# prompt_thinking_fixed_mass_conserving_cost_reasoning.py
import os
import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from glob import glob

from openai import OpenAI


# ==== CONFIGURATION ====

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# One script serves every model; set these per run.
MODEL_NAME = os.getenv("BEER_MODEL", "openai/gpt-oss-20b")
MODEL_TAG = os.getenv("BEER_MODEL_TAG", "oss20b")
N_SIM = 11

# Which sessions this process plays, e.g. BEER_SIM_IDS=6,7,8. Sessions are independent,
# so separate processes can take disjoint subsets.
_sim_ids = os.getenv("BEER_SIM_IDS")
SIM_IDS = (
    list(range(N_SIM))
    if _sim_ids is None
    else [int(x) for x in _sim_ids.split(",") if x.strip()]
)

# How many roles are asked to decide at once; see Pass B. 1 = one at a time.
ROLE_CONCURRENCY = int(os.getenv("BEER_ROLE_CONCURRENCY", "4"))

# Sampling temperature, e.g. BEER_TEMPERATURE=0.2.
TEMPERATURE = float(os.getenv("BEER_TEMPERATURE", "1"))
N_ROUNDS = 35
ANNOUNCED_GAME_LENGTH = 50
INITIAL_INVENTORY = 12

ROLES = ["Retailer", "Wholesaler", "Distributor", "Factory"]

UPSTREAM = {
    "Retailer": "Wholesaler",
    "Wholesaler": "Distributor",
    "Distributor": "Factory",
    "Factory": None,
}

DOWNSTREAM = {
    "Factory": "Distributor",
    "Distributor": "Wholesaler",
    "Wholesaler": "Retailer",
    "Retailer": None,
}

ORDER_DELAY = 2
SHIP_DELAY = 2
PROD_LEAD_TIME = 2

INITIAL_THROUGHPUT = 4
FORCED_WARMUP_ROUNDS = 4

# Local history shown to the agent, in rounds. None means the complete record: every round
# played so far, for both the orders observed and the orders placed. The companion arm sets
# this to 4, so the two differ only in this constant.
HISTORY_WINDOW = None

HISTORY_LABEL = (
    "all rounds" if HISTORY_WINDOW is None else f"last {HISTORY_WINDOW} rounds"
)


# ---- PRICING STRUCTURE ----

SELLING_PRICES = {
    "Retailer": 12.0,
    "Wholesaler": 8.0,
    "Distributor": 6.0,
    "Factory": 4.0,
}

BUYING_PRICES = {
    "Retailer": 8.0,
    "Wholesaler": 6.0,
    "Distributor": 4.0,
    "Factory": 2.0,
}


# ---- COST STRUCTURE ----

INVENTORY_HOLDING_COST = 0.5
BACKORDER_PENALTY = 1.0


# ---- CLIENT SETUP ----

# Retries are handled by the loop below rather than by the SDK.
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    timeout=180,
    max_retries=0,
)

# Keeps log lines intact when roles decide concurrently.
_print_lock = threading.Lock()


def log(message):
    with _print_lock:
        print(message, flush=True)


# ---- DEMAND PROFILE ----

customer_demand = [
    4, 4, 4, 4,
    8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8,
]


ROLE_CONTEXT = {
    "Retailer": "You are the Retailer. You buy from the Wholesaler and sell to consumers.",
    "Wholesaler": "You are the Wholesaler. You buy from the Distributor and sell to the Retailer.",
    "Distributor": "You are the Distributor. You buy from the Factory and sell to the Wholesaler.",
    "Factory": "You are the Factory. You produce beer and sell to the Distributor.",
}


# ---- UTILITY FUNCTIONS ----

def to_nonnegative_int(value, default=0):
    """
    Convert a value to a nonnegative integer.
    """
    try:
        value = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(0, value)


def parse_order_and_cost_reasoning(content):
    """
    Parse the LLM response without enforcing a rigid accounting template.

    Preferred response format:
        First line: integer order quantity, or ORDER: <integer>
        Remaining lines: prose cost/accounting rationale, with arithmetic if useful.

    Returns:
        order_qty: nonnegative integer
        cost_reasoning: all non-order text after the parsed order line
        raw_response: full raw model response
    """
    raw_response = content or ""
    stripped = raw_response.strip()

    order_qty = None
    order_line_index = None

    lines = [line.rstrip() for line in stripped.splitlines()]

    # Preferred format: ORDER: 8
    for idx, line in enumerate(lines):
        match = re.match(r"^\s*ORDER\s*:\s*(\d+)\b", line, re.IGNORECASE)
        if match:
            order_qty = int(match.group(1))
            order_line_index = idx
            break

    # Preferred simple format: first nonempty line is exactly an integer.
    if order_qty is None:
        for idx, line in enumerate(lines):
            line_clean = line.strip()
            if not line_clean:
                continue
            if re.fullmatch(r"\d+", line_clean):
                order_qty = int(line_clean)
                order_line_index = idx
                break

    # Fallback: "Line 1: 8", "1. 8", "Order quantity = 8", etc.
    if order_qty is None:
        for idx, line in enumerate(lines):
            match = re.match(
                r"^\s*(?:(?:line\s*)?1\s*[:.)-]|order(?:\s+quantity)?\s*[=:])\s*(\d+)\b",
                line,
                re.IGNORECASE,
            )
            if match:
                order_qty = int(match.group(1))
                order_line_index = idx
                break

    # Fallback: first integer in the response.
    if order_qty is None:
        for idx, line in enumerate(lines):
            match = re.search(r"\b(\d+)\b", line)
            if match:
                order_qty = int(match.group(1))
                order_line_index = idx
                break

    if order_qty is None:
        order_qty = 0

    order_qty = to_nonnegative_int(order_qty)

    # Keep everything except the one line used to parse the order.
    kept_lines = []
    for idx, line in enumerate(lines):
        if idx == order_line_index:
            continue
        kept_lines.append(line)

    cost_reasoning = "\n".join(kept_lines).strip()

    # If the model returned only one line, try to remove just the order token and keep any text after it.
    if not cost_reasoning and order_line_index is not None and order_line_index < len(lines):
        line = lines[order_line_index]
        line = re.sub(r"^\s*ORDER\s*:\s*\d+\b\s*[-:,.]?\s*", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^\s*\d+\b\s*[-:,.]?\s*", "", line)
        cost_reasoning = line.strip()

    return order_qty, cost_reasoning, raw_response


def get_player_prompt(role, info, round_number=None):
    context = ROLE_CONTEXT[role]

    on_hand_units = to_nonnegative_int(info.get("inventory", 0))
    backlog_units = to_nonnegative_int(info.get("backlog", 0))
    demand = to_nonnegative_int(info.get("order_received", 0))

    effective_inventory_units = on_hand_units - backlog_units

    if role == "Factory":
        pipeline = info.get("production_in_progress", [])
        pipeline_label = "Production pipeline"
    else:
        pipeline = info.get("shipments_in_transit", [])
        pipeline_label = "Inbound shipment pipeline"

    weekly_arrivals = [
        sum(to_nonnegative_int(s.get("qty", 0)) for s in pipeline if s.get("eta", 0) == w)
        for w in (1, 2)
    ]
    incoming_next_two = sum(weekly_arrivals)

    recent_demand_list = info.get("recent_demand", [])

    if len(recent_demand_list) >= 2:
        last_week_demand = to_nonnegative_int(recent_demand_list[-2])
    elif len(recent_demand_list) == 1:
        last_week_demand = to_nonnegative_int(recent_demand_list[-1])
    else:
        last_week_demand = 0

    last_order = to_nonnegative_int(info.get("last_order", 0))
    order_history = info.get("order_history", [])
    inventory_history = info.get("inventory_history", [])
    backlog_history = info.get("backlog_history", [])

    current_inventory_cost = on_hand_units * INVENTORY_HOLDING_COST
    current_backlog_cost = backlog_units * BACKORDER_PENALTY
    current_operating_cost = current_inventory_cost + current_backlog_cost

    decision_guidelines = (
        f"\n[DECISION PROCESS: ANCHORING AND ADJUSTMENT]\n"
        f"Make this decision as a bounded-rational local manager using a simple "
        f"anchoring-and-adjustment rule.\n\n"

        f"1. Start from a replacement anchor, but do not stop there.\n"
        f"   A natural first guess is to replace expected losses from stock, using recent "
        f"orders from your immediate customer as the most salient cue. This replacement "
        f"anchor is only a starting point. It is not a floor. If inventory is already too "
        f"high, simply replacing current demand would keep excess inventory from falling, "
        f"so the final order may be below recent demand, including zero.\n\n"

        f"2. Adjust toward the reference stock level anchored at the initial inventory.\n"
        f"   Use {INITIAL_INVENTORY} units as the simple reference level for effective "
        f"inventory, because the game began at that level and there is no reliable way to "
        f"calculate an optimal target. Compare effective inventory, defined as on-hand "
        f"inventory minus backlog, to this reference level.\n"
        f"   - If effective inventory is below {INITIAL_INVENTORY}, this creates upward "
        f"pressure on the order.\n"
        f"   - If effective inventory is above {INITIAL_INVENTORY}, this creates downward "
        f"pressure on the order.\n"
        f"   - The larger the excess above {INITIAL_INVENTORY}, the stronger the downward "
        f"pressure should be.\n"
        f"   Make only a partial adjustment, but allow a large excess stock position to "
        f"push the final order well below the demand anchor.\n\n"

        f"3. Account for the supply line, which is not given to you.\n"
        f"   Your supply line is the units you have already ordered but not yet "
        f"received. No total for it appears in the state below. The ETA pipeline shows "
        f"only the part your supplier has already shipped; anything you ordered that has "
        f"not been shipped yet appears nowhere, so your own order history is the only "
        f"record of it. Judge the supply line from those two sources and use it as a "
        f"rough, partial delayed-effect cue. Human decision makers often underweight "
        f"delayed pipeline effects, so do not mechanically subtract every unit you "
        f"believe is on the way. But if inventory is already above the reference level "
        f"and you judge the supply line to be large, this should further reduce the new "
        f"order.\n\n"

        f"4. Convert the adjusted pressure into a nonnegative integer order.\n"
        f"   The final order is allowed to be smaller than recent demand. It can be zero "
        f"when effective inventory is well above the reference level or when visible "
        f"incoming units make additional ordering feel unnecessary.\n\n"

        f"CURRENT SITUATION THIS ROUND:\n"
        f"- This state is after you received any arriving shipment and after you filled as much "
        f"downstream demand/backlog as possible this round.\n"
        f"- Your on-hand inventory: {on_hand_units} units\n"
        f"- Your backlog: {backlog_units} unfilled units\n"
        f"- Effective inventory: {effective_inventory_units} units "
        f"({on_hand_units} inventory - {backlog_units} backlog)\n"
        f"- Visible ETA pipeline, shown only as a breakdown/subset of that total:\n"
        f"  ETA 1: {weekly_arrivals[0]} units\n"
        f"  ETA 2: {weekly_arrivals[1]} units\n"
        f"- Demand/order received this round: {demand} units\n"
        f"- Last week's demand/order received: {last_week_demand} units\n"
        f"- Recent incoming orders you observed: {recent_demand_list}\n"
        f"- Last order you placed: {last_order} units\n"
        f"- Your order history ({HISTORY_LABEL}): {order_history}\n"
        f"- Your inventory history ({HISTORY_LABEL}): {inventory_history}\n"
        f"- Your backlog history ({HISTORY_LABEL}): {backlog_history}\n"
        f"- Inventory holding cost this round: {on_hand_units} * ${INVENTORY_HOLDING_COST} = ${current_inventory_cost:.2f}\n"
        f"- Backorder penalty this round: {backlog_units} * ${BACKORDER_PENALTY} = ${current_backlog_cost:.2f}\n"
    )

    prompt = (
        f"{context}\n\n"
        f"This is the Beer Game simulation. It lasts {ANNOUNCED_GAME_LENGTH} rounds. You are currently in round {round_number}.\n"
        f"Your objective is to manage future inventory/backlog costs using only local information. "
        f"Current demand has already been filled or backlogged before this decision; the order you place now "
        f"will affect future inventory only after the order and shipping/production delays. "
        f"The cost of holding inventory is ${INVENTORY_HOLDING_COST} per unit per round. "
        f"The cost of backorders is ${BACKORDER_PENALTY} per unit per round.\n"
        f"There is a {ORDER_DELAY}-round order lead time and a {SHIP_DELAY}-round shipping lead time "
        f"or production lead time if you are the Factory.\n"
        f"You are part of a supply chain and make decisions from your own local information. "
        f"While you minimize your own costs, also consider the impact of your ordering on the rest "
        f"of the team—minimizing total team cost is beneficial.\n"
        f"\n[CURRENT STATE]\n"
        f"{decision_guidelines}\n"
        f"\nRespond with the order quantity on the first line so it can be recorded reliably. "
        f"After that, give a brief heuristic rationale in one or two sentences. Explain how the "
        f"replacement anchor was adjusted upward or downward by the stock gap and visible pipeline. "
        f"Do not treat current demand as a minimum order.\n\n"
        "Example format:\n"
        "<integer order quantity>\n"
        "<your rationale here>"
    )

    return prompt


def openrouter_llm_decision(prompt, model=MODEL_NAME, max_retries=8, label=""):
    """
    Call OpenRouter API and parse the returned order quantity plus prose cost/accounting rationale.
    Retries up to max_retries times if the API returns None content.
    """
    import time

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=50000,
                max_completion_tokens=50000,
                reasoning_effort="low",
            )

            content = response.choices[0].message.content
            if content is None:
                raise ValueError("API returned None content")
            content = content.strip()

        except Exception as e:
            if attempt < max_retries:
                wait_time = min(60.0, 3.0 * 2 ** (attempt - 1)) + random.uniform(0, 3)
                log(f"⚠️ Attempt {attempt}/{max_retries}{label} failed: {e}\n"
                    f"   Retrying in {wait_time:.0f}s...")
                time.sleep(wait_time)
                continue
            else:
                log(f"❌ All {max_retries} attempts{label} failed. Raising error.")
                raise

        break

    log(f"🧠 RAW FINAL RESPONSE{label}:\n" + "-" * 40 + f"\n{content}\n" + "-" * 40)

    order_qty, cost_reasoning, raw_response = parse_order_and_cost_reasoning(content)

    if order_qty is None:
        print("⚠️ Warning: Failed to parse a valid order quantity from the response.")
        order_qty = 0

    return to_nonnegative_int(order_qty), cost_reasoning, raw_response


def calculate_comprehensive_costs(history, role, player):
    """
    Calculate detailed costs from player history and player object.

    Inventory is on-hand physical inventory.
    Backlog is stored separately.
    There is no negative inventory.
    """
    if not history:
        return {
            "inventory_cost": 0.0,
            "backorder_cost": 0.0,
            "operating_costs": 0.0,
            "total_revenue": 0.0,
            "total_cogs": 0.0,
            "total_costs": 0.0,
            "total_economic_costs": 0.0,
            "net_profit": 0.0,
        }

    inventory_cost = 0.0
    backorder_cost = 0.0

    for h in history:
        inventory_cost += to_nonnegative_int(h.get("inventory", 0)) * INVENTORY_HOLDING_COST
        backorder_cost += to_nonnegative_int(h.get("backlog", 0)) * BACKORDER_PENALTY

    operating_costs = inventory_cost + backorder_cost

    total_revenue = float(player.total_revenue)
    total_cogs = float(player.total_cogs)
    total_economic_costs = operating_costs + total_cogs

    net_profit = total_revenue - total_economic_costs

    return {
        "inventory_cost": inventory_cost,
        "backorder_cost": backorder_cost,
        "operating_costs": operating_costs,

        # Kept for backward compatibility with old aggregation code.
        # This is holding + backlog cost, not including COGS.
        "total_costs": operating_costs,

        "total_revenue": total_revenue,
        "total_cogs": total_cogs,
        "total_economic_costs": total_economic_costs,
        "net_profit": net_profit,
    }


# ---- BEER GAME LOGIC ----

def make_seeded_delay_pipeline(delay_length, qty=INITIAL_THROUGHPUT):
    """
    Create a delay pipeline already in equilibrium.

    In this simulation, pipelines are advanced at the beginning of each round.
    Therefore:
    - eta=1 arrives in the current round after advance_pipelines()
    - eta=2 arrives in the next round after advance_pipelines()

    With ORDER_DELAY = 2 or SHIP_DELAY = 2, this creates two delay boxes,
    each containing INITIAL_THROUGHPUT units.
    """
    return [
        {"qty": to_nonnegative_int(qty), "eta": eta}
        for eta in range(1, int(delay_length) + 1)
    ]


class Player:
    def __init__(self, name):
        self.name = name

        # Physical on-hand inventory only. Never negative.
        self.inventory = INITIAL_INVENTORY

        # Unfilled downstream orders. Tracked separately from inventory.
        self.backlog = 0

        self.history = []
        self.last_orders = []
        self.outstanding_to_supplier = 0

        # Orders from downstream that will arrive after ORDER_DELAY.
        self.order_pipeline = []

        # Inbound shipments from supplier, or production-in-progress for Factory.
        self.shipment_pipeline = []

        self.total_revenue = 0.0
        self.total_costs = 0.0       # holding + backlog costs only
        self.total_cogs = 0.0

    def advance_pipelines(self):
        for s in self.order_pipeline:
            s["eta"] -= 1

        for s in self.shipment_pipeline:
            s["eta"] -= 1

    def pop_arriving_order(self):
        arriving = [s for s in self.order_pipeline if s["eta"] <= 0]
        self.order_pipeline = [s for s in self.order_pipeline if s["eta"] > 0]
        return sum(to_nonnegative_int(s["qty"]) for s in arriving)

    def pop_arriving_shipment(self):
        arriving = [s for s in self.shipment_pipeline if s["eta"] <= 0]
        self.shipment_pipeline = [s for s in self.shipment_pipeline if s["eta"] > 0]
        return sum(to_nonnegative_int(s["qty"]) for s in arriving)

    def add_order(self, qty, eta):
        qty = to_nonnegative_int(qty)
        eta = int(eta)

        if qty > 0:
            self.order_pipeline.append({"qty": qty, "eta": eta})

    def add_shipment(self, qty, eta):
        qty = to_nonnegative_int(qty)
        eta = int(eta)

        if qty > 0:
            self.shipment_pipeline.append({"qty": qty, "eta": eta})

    def receive_shipment(self, qty):
        """
        Receive actual physical shipment and update inventory and COGS.
        """
        qty = to_nonnegative_int(qty)

        if qty <= 0:
            return

        self.outstanding_to_supplier = max(0, self.outstanding_to_supplier - qty)

        self.inventory += qty

        purchase_cost = qty * BUYING_PRICES[self.name]
        self.total_cogs += purchase_cost

    def fulfill_order(self, incoming_order):
        """
        Fulfill downstream demand/orders in a mass-conserving way.

        Correct mechanics:
        1. Incoming downstream order is added to backlog.
        2. Actual shipped quantity is min(on-hand inventory, backlog).
        3. Inventory falls only by actual shipped quantity.
        4. Backlog falls only by actual shipped quantity.
        5. Returned shipped quantity is what enters the downstream shipment pipeline.
        """
        incoming_order = to_nonnegative_int(incoming_order)

        self.backlog += incoming_order

        shipped = min(self.inventory, self.backlog)

        self.inventory -= shipped
        self.backlog -= shipped

        revenue = 0.0
        if shipped > 0:
            revenue = shipped * SELLING_PRICES[self.name]
            self.total_revenue += revenue

        return shipped, revenue

    def calculate_period_costs(self):
        """
        Holding cost is charged on nonnegative physical inventory.
        Backorder penalty is charged on explicit backlog.
        """
        inventory_cost = self.inventory * INVENTORY_HOLDING_COST
        backorder_cost = self.backlog * BACKORDER_PENALTY

        period_costs = inventory_cost + backorder_cost

        self.total_costs += period_costs

        return period_costs, inventory_cost, backorder_cost

    def place_order(self, prompt_info):
        prompt = get_player_prompt(self.name, prompt_info, round_number=prompt_info.get("round_number"))

        log(f"\n--- {self.name} Prompt ---\n{prompt}\n")

        order_qty, cost_reasoning, raw_response = openrouter_llm_decision(
            prompt, label=f" [{self.name}]"
        )

        log(f"{self.name} decides to order: {order_qty}\n"
            f"Cost/accounting reasoning:\n{cost_reasoning}")

        return order_qty, cost_reasoning, raw_response


def seed_initial_equilibrium(players):
    """
    Initialize the Beer Game in Sterman's equilibrium state.

    Sterman equilibrium:
    - each role starts with 12 units of inventory
    - equilibrium throughput is 4 units/week
    - each order delay contains an order slip for 4
    - each shipping/production delay contains 4 units
    - customer demand is initially 4 units/week
    """

    # Reset all state variables cleanly.
    for role, player in players.items():
        player.inventory = INITIAL_INVENTORY
        player.backlog = 0

        player.history = []
        player.last_orders = []

        player.order_pipeline = []
        player.shipment_pipeline = []

        player.total_revenue = 0.0
        player.total_costs = 0.0
        player.total_cogs = 0.0

        player.outstanding_to_supplier = 0

    # Order slips already in the delay boxes. The Retailer takes exogenous customer demand
    # directly and so has no order pipeline.
    for role in ["Wholesaler", "Distributor", "Factory"]:
        players[role].order_pipeline = make_seeded_delay_pipeline(
            ORDER_DELAY,
            INITIAL_THROUGHPUT,
        )

    # Beer already in shipping delays for roles that receive from an upstream supplier.
    # These are physical shipments already on their way.
    for role in ["Retailer", "Wholesaler", "Distributor"]:
        players[role].shipment_pipeline = make_seeded_delay_pipeline(
            SHIP_DELAY,
            INITIAL_THROUGHPUT,
        )

        # From this role's perspective, both order-delay units and shipping-delay units
        # are orders already placed but not yet physically received.
        player_supply_line = (ORDER_DELAY + SHIP_DELAY) * INITIAL_THROUGHPUT
        players[role].outstanding_to_supplier = player_supply_line

    # Factory has production already in progress.
    # In this code, factory production uses shipment_pipeline as production_in_progress.
    players["Factory"].shipment_pipeline = make_seeded_delay_pipeline(
        PROD_LEAD_TIME,
        INITIAL_THROUGHPUT,
    )

    # Factory's own supply line is production requested but not yet completed.
    players["Factory"].outstanding_to_supplier = (
        PROD_LEAD_TIME * INITIAL_THROUGHPUT
    )


def run_beer_game(sim_id=0):
    players = {role: Player(role) for role in ROLES}
    seed_initial_equilibrium(players)

    for round_idx in range(N_ROUNDS):
        print(f"\n================ ROUND {round_idx + 1} ================\n")

        orders_arriving = {}
        shipments_arriving = {}

        # Advance pipelines and collect arrivals.
        for role in ROLES:
            players[role].advance_pipelines()

            if role == "Retailer":
                orders_arriving[role] = customer_demand[round_idx]
            else:
                orders_arriving[role] = players[role].pop_arriving_order()

            shipments_arriving[role] = players[role].pop_arriving_shipment()

        new_orders = {}

        # Actual physical shipments made by each role this round.
        # These, not order quantities, are what should be delivered downstream.
        shipped_to_downstream = {}

        # Pass A: the physical flows, none of which depend on a decision made this round.
        round_state = {}

        for role in ROLES:
            player = players[role]

            # 1. Receive actual inbound shipment/production.
            player.receive_shipment(shipments_arriving[role])

            # 2. Fulfill downstream order/backlog using physical inventory.
            order_from_downstream = orders_arriving[role]
            shipped_qty, revenue = player.fulfill_order(order_from_downstream)
            shipped_to_downstream[role] = shipped_qty

            # 3. Charge holding/backlog costs after shipment decision.
            period_costs, inventory_cost, backorder_cost = player.calculate_period_costs()

            round_state[role] = {
                "order_from_downstream": order_from_downstream,
                "shipped_qty": shipped_qty,
                "revenue": revenue,
                "period_costs": period_costs,
                "inventory_cost": inventory_cost,
                "backorder_cost": backorder_cost,
                "net_profit": player.total_revenue - player.total_costs - player.total_cogs,
            }

        # Shipments enter the downstream pipeline before anyone decides, so the ETA 2 slot
        # holds what the supplier dispatched this round. Arrival timing is unchanged.
        for role in ROLES:
            dn = DOWNSTREAM[role]

            if dn is not None:
                players[dn].add_shipment(shipped_to_downstream[role], SHIP_DELAY)

        # Pass B: the decisions. Each role reads only its own player and every cross-role
        # effect happens in steps 6 and 8, so the four calls can be issued together and
        # the results applied in ROLES order.
        prompt_infos = {}

        for role in ROLES:
            player = players[role]
            state = round_state[role]

            order_from_downstream = state["order_from_downstream"]
            shipped_qty = state["shipped_qty"]
            revenue = state["revenue"]
            period_costs = state["period_costs"]
            inventory_cost = state["inventory_cost"]
            backorder_cost = state["backorder_cost"]
            net_profit = state["net_profit"]

            # Both series honour HISTORY_WINDOW: the orders observed end with this round,
            # and the orders placed are those the agent has made. With HISTORY_WINDOW set
            # to None neither is trimmed, so the agent sees the complete record.
            recent_demand = (
                [h["order_received"] for h in player.history] + [order_from_downstream]
            )
            recent_own_orders = list(player.last_orders)

            # The inventory and backlog series end with this round's post-fulfilment
            # figures, the same two numbers reported above them as the current state.
            inventory_history = (
                [h["inventory"] for h in player.history] + [player.inventory]
            )
            backlog_history = (
                [h["backlog"] for h in player.history] + [player.backlog]
            )

            if HISTORY_WINDOW is not None:
                recent_demand = recent_demand[-HISTORY_WINDOW:]
                recent_own_orders = recent_own_orders[-HISTORY_WINDOW:]
                inventory_history = inventory_history[-HISTORY_WINDOW:]
                backlog_history = backlog_history[-HISTORY_WINDOW:]

            if role == "Factory":
                prompt_infos[role] = {
                    "role": role,
                    "inventory": player.inventory,
                    "backlog": player.backlog,
                    "order_received": order_from_downstream,
                    "production_in_progress": [dict(s) for s in player.shipment_pipeline],
                    "total_revenue": player.total_revenue,
                    "total_costs": player.total_costs,
                    "total_cogs": player.total_cogs,
                    "net_profit": net_profit,
                    "recent_demand": recent_demand,
                    "last_order": player.last_orders[-1] if player.last_orders else 0,
                    "order_history": recent_own_orders,
                    "inventory_history": inventory_history,
                    "backlog_history": backlog_history,
                    "outstanding_to_supplier": player.outstanding_to_supplier,
                    "round_number": round_idx + 1,
                }
            else:
                prompt_infos[role] = {
                    "role": role,
                    "inventory": player.inventory,
                    "backlog": player.backlog,
                    "order_received": order_from_downstream,
                    "shipments_in_transit": [dict(s) for s in player.shipment_pipeline],
                    "total_revenue": player.total_revenue,
                    "total_costs": player.total_costs,
                    "total_cogs": player.total_cogs,
                    "net_profit": net_profit,
                    "recent_demand": recent_demand,
                    "last_order": player.last_orders[-1] if player.last_orders else 0,
                    "order_history": recent_own_orders,
                    "inventory_history": inventory_history,
                    "backlog_history": backlog_history,
                    "outstanding_to_supplier": player.outstanding_to_supplier,
                    "round_number": round_idx + 1,
                }

        # 4. Decide new order/production quantity. Rounds 1-4 are Sterman warm-up practice at
        # the equilibrium order, so no model is called.
        decisions = {}

        if round_idx < FORCED_WARMUP_ROUNDS:
            for role in ROLES:
                decisions[role] = (
                    INITIAL_THROUGHPUT,
                    (
                        f"Forced warm-up equilibrium round {round_idx + 1}: "
                        f"the protocol fixes the order at {INITIAL_THROUGHPUT} units "
                        f"to maintain the initial 4-unit throughput."
                    ),
                    (
                        f"{INITIAL_THROUGHPUT}\n"
                        f"Forced warm-up equilibrium order; no LLM decision was made."
                    ),
                    "forced_warmup",
                )
        else:

            def decide(role):
                order_qty, cost_reasoning, raw_response = players[role].place_order(
                    prompt_infos[role]
                )
                return role, (order_qty, cost_reasoning, raw_response, "llm")

            if ROLE_CONCURRENCY > 1:
                with ThreadPoolExecutor(max_workers=ROLE_CONCURRENCY) as pool:
                    for role, decision in pool.map(decide, ROLES):
                        decisions[role] = decision
            else:
                for role in ROLES:
                    decisions[role] = decide(role)[1]

        for role in ROLES:
            player = players[role]
            state = round_state[role]

            order_from_downstream = state["order_from_downstream"]
            shipped_qty = state["shipped_qty"]
            revenue = state["revenue"]
            period_costs = state["period_costs"]
            inventory_cost = state["inventory_cost"]
            backorder_cost = state["backorder_cost"]
            net_profit = state["net_profit"]

            order_qty, cost_reasoning, raw_response, decision_source = decisions[role]

            new_orders[role] = order_qty
            player.last_orders.append(order_qty)
            player.outstanding_to_supplier += order_qty

            # 5. Record round-level history.
            player.history.append({
                "round": round_idx + 1,

                "inventory": player.inventory,
                "backlog": player.backlog,

                "order_received": order_from_downstream,
                "order_placed": order_qty,

                "shipment_received": shipments_arriving[role],
                "shipped_qty": shipped_qty,

                "revenue": revenue,

                "inventory_cost": inventory_cost,
                "backorder_cost": backorder_cost,
                "period_costs": period_costs,

                "total_revenue": player.total_revenue,
                "total_costs": player.total_costs,
                "total_cogs": player.total_cogs,
                "net_profit": net_profit,

                "decision_source": decision_source,
                "cost_reasoning": cost_reasoning,
                # Backward-compatible alias for earlier code that expected a justification field.
                "justification": cost_reasoning,
                "raw_llm_response": raw_response,
            })

        # 6. Add new orders to upstream players' order pipelines.
        # These are information/order flows, not physical shipment flows.
        for role in ROLES:
            up = UPSTREAM[role]

            if up is not None:
                players[up].add_order(new_orders[role], ORDER_DELAY)

        # 8. Factory production creates new units after production lead time.
        players["Factory"].add_shipment(new_orders["Factory"], PROD_LEAD_TIME)

    single_orders = {
        role: [h["order_placed"] for h in players[role].history]
        for role in ROLES
    }

    single_inventories = {
        role: [h["inventory"] for h in players[role].history]
        for role in ROLES
    }

    single_backlogs = {
        role: [h["backlog"] for h in players[role].history]
        for role in ROLES
    }

    single_shipments = {
        role: [h["shipped_qty"] for h in players[role].history]
        for role in ROLES
    }

    single_cost_reasoning = {
        role: [h["cost_reasoning"] for h in players[role].history]
        for role in ROLES
    }

    financial_results = {
        role: calculate_comprehensive_costs(players[role].history, role, players[role])
        for role in ROLES
    }

    history = {
        role: [
            {
                "round": h["round"],

                "order_received": h["order_received"],
                "order_placed": h["order_placed"],

                "shipment_received": h["shipment_received"],
                "shipped_qty": h["shipped_qty"],

                "inventory": h["inventory"],
                "backlog": h["backlog"],

                "inventory_cost": h["inventory_cost"],
                "backorder_cost": h["backorder_cost"],
                "period_costs": h["period_costs"],

                "revenue": h["revenue"],
                "total_revenue": h["total_revenue"],
                "total_costs": h["total_costs"],
                "total_cogs": h["total_cogs"],
                "net_profit": h["net_profit"],

                "decision_source": h.get("decision_source", "llm"),
                "cost_reasoning": h["cost_reasoning"],
                "justification": h["justification"],
                "raw_llm_response": h["raw_llm_response"],
            }
            for h in players[role].history
        ]
        for role in ROLES
    }

    return {
        "sim_id": sim_id,
        "orders": single_orders,
        "inventories": single_inventories,
        "backlogs": single_backlogs,
        "shipments": single_shipments,
        "cost_reasoning": single_cost_reasoning,
        "financial_results": financial_results,
        "history": history,
    }


def sanity_check_warmup_equilibrium(result):
    """
    Basic checks that the Sterman warm-up protocol is working.
    This should be called on one simulation result during debugging.
    """
    expected_orders = [INITIAL_THROUGHPUT] * FORCED_WARMUP_ROUNDS

    for role in ROLES:
        actual_orders = result["orders"][role][:FORCED_WARMUP_ROUNDS]
        assert actual_orders == expected_orders, (
            f"{role} warm-up orders wrong: expected {expected_orders}, "
            f"got {actual_orders}"
        )

        warmup_history = result["history"][role][:FORCED_WARMUP_ROUNDS]

        for h in warmup_history:
            assert h["decision_source"] == "forced_warmup", (
                f"{role} round {h['round']} was not marked forced_warmup"
            )
            assert h["order_placed"] == INITIAL_THROUGHPUT, (
                f"{role} round {h['round']} order was not {INITIAL_THROUGHPUT}"
            )
            assert h["backlog"] == 0, (
                f"{role} round {h['round']} backlog should be 0, got {h['backlog']}"
            )
            assert h["inventory"] == INITIAL_INVENTORY, (
                f"{role} round {h['round']} inventory should remain "
                f"{INITIAL_INVENTORY}, got {h['inventory']}"
            )


def run_all_simulations():
    OUTPUT_DIR = f"{MODEL_TAG}-prompt"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for sim in SIM_IDS:
        sim_path = os.path.join(OUTPUT_DIR, f"sim_{sim}.json")

        if os.path.exists(sim_path):
            print(f"✅ Simulation {sim} already exists. Skipping.")
            continue

        print(f"\n🔄 Running simulation {sim}...\n")

        max_sim_retries = 3
        for sim_attempt in range(1, max_sim_retries + 1):
            try:
                result = run_beer_game(sim_id=sim)

                if sim == 0:
                    sanity_check_warmup_equilibrium(result)

                with open(sim_path, "w") as f:
                    json.dump(result, f, indent=2)

                print(f"✅ Simulation {sim} completed and saved.")
                break

            except Exception as e:
                print(f"❌ Error in simulation {sim} (attempt {sim_attempt}/{max_sim_retries}): {e}")
                if sim_attempt < max_sim_retries:
                    import time
                    wait_time = 5 * sim_attempt
                    print(f"   Retrying simulation {sim} in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Simulation {sim} failed after {max_sim_retries} attempts. Skipping.")

    # Only the process that finds every session on disk writes the aggregate.
    on_disk = glob(os.path.join(OUTPUT_DIR, "sim_*.json"))
    if len(on_disk) < N_SIM:
        print(f"\n{len(on_disk)}/{N_SIM} sessions on disk. Leaving the aggregate to "
              f"whichever process finishes last.")
        return None

    all_json_paths = sorted(glob(os.path.join(OUTPUT_DIR, "sim_*.json")))

    all_results = []
    for path in all_json_paths:
        with open(path, "r") as f:
            all_results.append(json.load(f))

    all_orders = {role: [] for role in ROLES}
    all_inventories = {role: [] for role in ROLES}
    all_backlogs = {role: [] for role in ROLES}
    all_shipments = {role: [] for role in ROLES}
    all_cost_reasoning = {role: [] for role in ROLES}

    financial_accum = {
        role: {
            "total_revenue": [],
            "total_costs": [],
            "total_cogs": [],
            "total_economic_costs": [],
            "net_profit": [],
        }
        for role in ROLES
    }

    for result in all_results:
        for role in ROLES:
            all_orders[role].append(result["orders"][role])
            all_inventories[role].append(result["inventories"][role])
            all_backlogs[role].append(result["backlogs"][role])
            all_shipments[role].append(result["shipments"][role])
            all_cost_reasoning[role].append(result["cost_reasoning"][role])

            financial_accum[role]["total_revenue"].append(
                result["financial_results"][role]["total_revenue"]
            )
            financial_accum[role]["total_costs"].append(
                result["financial_results"][role]["total_costs"]
            )
            financial_accum[role]["total_cogs"].append(
                result["financial_results"][role]["total_cogs"]
            )
            financial_accum[role]["total_economic_costs"].append(
                result["financial_results"][role]["total_economic_costs"]
            )
            financial_accum[role]["net_profit"].append(
                result["financial_results"][role]["net_profit"]
            )

    aggregate = {
        "all_orders": all_orders,
        "all_inventories": all_inventories,
        "all_backlogs": all_backlogs,
        "all_shipments": all_shipments,
        "all_cost_reasoning": all_cost_reasoning,
        "financial_accum": financial_accum,
    }

    aggregate_path = os.path.join(OUTPUT_DIR, "aggregate_results.json")
    tmp_path = f"{aggregate_path}.{os.getpid()}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    os.replace(tmp_path, aggregate_path)

    print(f"\n✅ Aggregate results saved to {aggregate_path}")

    return aggregate


if __name__ == "__main__":
    run_all_simulations()
