# Digit Capture Plan

- Generated: `2026-02-25T11:42:28.318971+00:00`
- Target train samples per digit: `12`
- Priority digits: `4,5,6,9`
- Seed label for examples: `2314`

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
  Suggested reading labels to target: `4314`, `2414`, `2344`, `2314`, `4414`, `4344`, `2444`, `4444`

- [ ] Digit `5`: collect at least `12` additional train occurrences.
  Current train count: `0`; target: `12`.
  Suggested reading labels to target: `5314`, `2514`, `2354`, `2315`, `5514`, `5354`, `5315`, `2554`, `2515`, `2355`

- [ ] Digit `6`: collect at least `12` additional train occurrences.
  Current train count: `0`; target: `12`.
  Suggested reading labels to target: `6314`, `2614`, `2364`, `2316`, `6614`, `6364`, `6316`, `2664`, `2616`, `2366`

- [ ] Digit `9`: collect at least `9` additional train occurrences.
  Current train count: `3`; target: `12`.
  Suggested reading labels to target: `9314`, `2914`, `2394`, `2319`, `9914`, `9394`, `9319`, `2994`, `2919`, `2399`

## QA Loop

- After adding labels, rebuild dataset and run `python validate_digit_dataset.py` before training.