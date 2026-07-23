# Digit Capture Plan

- Generated: `2026-07-22T21:43:25.242485+00:00`
- Target train samples per digit: `12`
- Priority digits: `4,5,6,9`
- Seed label for examples: `2347`

## Coverage Snapshot

| Digit | Train | Val | Test | Total | Train Deficit |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 5 | 0 | 0 | 5 | 7 |
| 1 | 14 | 0 | 1 | 15 | 0 |
| 2 | 45 | 0 | 2 | 47 | 0 |
| 3 | 45 | 0 | 1 | 46 | 0 |
| 4 | 7 | 0 | 0 | 7 | 5 |
| 5 | 4 | 0 | 0 | 4 | 8 |
| 6 | 4 | 0 | 0 | 4 | 8 |
| 7 | 7 | 0 | 0 | 7 | 5 |
| 8 | 6 | 0 | 0 | 6 | 6 |
| 9 | 3 | 0 | 0 | 3 | 9 |

## Priority Checklist

- [ ] Digit `4`: collect at least `5` additional train occurrences.
  Current train count: `7`; target: `12`.
  Suggested reading labels to target: `4347`, `2447`, `2347`, `2344`, `4447`, `4344`, `2444`, `4444`

- [ ] Digit `5`: collect at least `8` additional train occurrences.
  Current train count: `4`; target: `12`.
  Suggested reading labels to target: `5347`, `2547`, `2357`, `2345`, `5547`, `5357`, `5345`, `2557`, `2545`, `2355`

- [ ] Digit `6`: collect at least `8` additional train occurrences.
  Current train count: `4`; target: `12`.
  Suggested reading labels to target: `6347`, `2647`, `2367`, `2346`, `6647`, `6367`, `6346`, `2667`, `2646`, `2366`

- [ ] Digit `9`: collect at least `9` additional train occurrences.
  Current train count: `3`; target: `12`.
  Suggested reading labels to target: `9347`, `2947`, `2397`, `2349`, `9947`, `9397`, `9349`, `2997`, `2949`, `2399`

## QA Loop

- After adding labels, rebuild dataset and run `python validate_digit_dataset.py` before training.