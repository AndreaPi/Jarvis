# Digit Capture Plan

- Generated: `2026-02-19T21:59:50.628873+00:00`
- Target train samples per digit: `12`
- Priority digits: `4,5,6,9`
- Seed label for examples: `2311`

## Coverage Snapshot

| Digit | Train | Val | Test | Total | Train Deficit |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 3 | 1 | 1 | 5 | 9 |
| 1 | 3 | 4 | 1 | 8 | 9 |
| 2 | 9 | 3 | 1 | 13 | 3 |
| 3 | 3 | 3 | 1 | 7 | 9 |
| 4 | 1 | 0 | 0 | 1 | 11 |
| 5 | 0 | 0 | 0 | 0 | 12 |
| 6 | 0 | 0 | 0 | 0 | 12 |
| 7 | 4 | 0 | 0 | 4 | 8 |
| 8 | 2 | 1 | 0 | 3 | 10 |
| 9 | 3 | 0 | 0 | 3 | 9 |

## Priority Checklist

- [ ] Digit `4`: collect at least `11` additional train occurrences.
  Current train count: `1`; target: `12`.
  Suggested reading labels to target: `4311`, `2411`, `2341`, `2314`, `4411`, `4341`, `4314`, `2441`, `2414`, `2344`

- [ ] Digit `5`: collect at least `12` additional train occurrences.
  Current train count: `0`; target: `12`.
  Suggested reading labels to target: `5311`, `2511`, `2351`, `2315`, `5511`, `5351`, `5315`, `2551`, `2515`, `2355`

- [ ] Digit `6`: collect at least `12` additional train occurrences.
  Current train count: `0`; target: `12`.
  Suggested reading labels to target: `6311`, `2611`, `2361`, `2316`, `6611`, `6361`, `6316`, `2661`, `2616`, `2366`

- [ ] Digit `9`: collect at least `9` additional train occurrences.
  Current train count: `3`; target: `12`.
  Suggested reading labels to target: `9311`, `2911`, `2391`, `2319`, `9911`, `9391`, `9319`, `2991`, `2919`, `2399`

## QA Loop

- After adding labels, rebuild dataset and run `python validate_digit_dataset.py` before training.