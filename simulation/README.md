# LLM agents in the Beer Distribution Game

Simulation code for two setups, three conditions each.

```
full_state/           one API call per decision; history is in the prompt
persistent_session/   one conversation per role; history is the transcript
  baseline_openrouter.py      no cognitive framing
  prompt_openrouter.py        anchoring-and-adjustment guidance
  prompt_alt_openrouter.py    cognitive-load framing
```

## Run

```bash
pip install openai
export OPENROUTER_API_KEY=...        # openrouter.ai/keys
```

```bash
BEER_MODEL=openai/gpt-oss-20b BEER_MODEL_TAG=oss20b-fs python3 -u full_state/baseline_openrouter.py
BEER_MODEL=openai/gpt-oss-20b BEER_MODEL_TAG=oss20b-fs python3 -u full_state/prompt_openrouter.py
BEER_MODEL=openai/gpt-oss-20b BEER_MODEL_TAG=oss20b-fs python3 -u full_state/prompt_alt_openrouter.py

BEER_MODEL=openai/gpt-oss-20b BEER_MODEL_TAG=oss20b-se python3 -u persistent_session/baseline_openrouter.py
BEER_MODEL=openai/gpt-oss-20b BEER_MODEL_TAG=oss20b-se python3 -u persistent_session/prompt_openrouter.py
BEER_MODEL=openai/gpt-oss-20b BEER_MODEL_TAG=oss20b-se python3 -u persistent_session/prompt_alt_openrouter.py
```

Each script plays 11 sessions of 35 rounds and writes `<tag>-<condition>/sim_N.json` plus
`aggregate_results.json`. The conditions are independent and can run in parallel.
Re-running skips sessions already on disk, so an interrupted run resumes.

Give the two setups different tags — both write to `<tag>-baseline/`, `<tag>-prompt/`,
`<tag>-alt/`.

Python 3.9+. Roughly 1,400 model calls per condition; wall time depends entirely on the
model.

## Environment variables

All operational — none change the game or the prompts.

| variable | default | applies to |
|---|---|---|
| `OPENROUTER_API_KEY` | — | both, required |
| `BEER_MODEL` | `openai/gpt-oss-20b` | both |
| `BEER_MODEL_TAG` | see script | both, prefixes the output directories |
| `BEER_TEMPERATURE` | `1` | `full_state` |
| `BEER_SIM_IDS` | all 11 | `full_state`, e.g. `0,1,2` to split sessions across processes |
| `BEER_ROLE_CONCURRENCY` | `4` | `full_state`, roles asked to decide at once; `1` for one at a time |

`persistent_session` uses temperature 1 and one role at a time. Both setups pass
`reasoning_effort="low"` as a literal in the API call.

## Parameters

Identical across all six scripts.

| | |
|---|---|
| roles | Retailer, Wholesaler, Distributor, Factory |
| sessions × rounds | 11 × 35, announced to the agent as 50 |
| forced warm-up | rounds 1-4 at 4 units, no model call |
| customer demand | 4 units, stepping to 8 at round 5 |
| initial inventory | 12 units, backlog 0 |
| order delay | 2 rounds |
| shipping / production delay | 2 rounds |
| holding / backorder cost | $0.50 / $1.00 per unit per round |

Inventory is non-negative with backlog tracked separately, and units are conserved.
Order-to-receipt is 4 rounds, as in Sterman (1989).

## What the agent sees

Its own inventory, backlog, demand received, and the shipments arriving in the next two
rounds. The ETA slots cover only what the supplier
has already shipped, so units ordered within the last two rounds are not visible anywhere
and must be inferred from the agent's own past orders. The Factory is its own supplier, so
its ETA 2 slot is always empty.

`full_state` adds four history arrays to every prompt — orders observed, orders placed,
inventory, backlog. `persistent_session` omits them, because the previous rounds are already
in the conversation.

Baseline and heuristic share an identical state block. The cognitive-load condition restates
the same quantities in experiential language. No condition sees a quantity another does not.

## Output

Per role and per round: order placed, order received, shipment received, quantity shipped,
inventory, backlog, costs, revenue, the model's rationale, and the raw response — enough to
recompute every reported quantity and check the conservation identities.

## Notes

- Temperature is 1 and the endpoints accept no seed, so re-running reproduces distributions,
  not individual trajectories.
- OpenRouter may route one model id to different providers at different times.
