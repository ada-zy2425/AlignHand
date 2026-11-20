# Split lists (IDs only)

One JSON object per line; fields are identifiers only (group key, sequence id, frame id). No annotations, coordinates or errors.

- `dexycb_fit`: 20329 entries
- `ho3d_fit`: 4056 entries
- `hanco_fit`: 68127 entries
- `dexycb_dev`: 1412 entries
- `ho3d_dev`: 3904 entries
- `hanco_a3_dev`: 7084 entries
- `dexycb_test`: 4951 entries
- `hanco_test_ust`: 20198 entries
- `ho3d_eval_mv_conf`: 3178 entries
- `ho3d_mv_poemtest`: 2706 entries

- `ho3d_mv_poemtest_excluded.json`: the 4 degenerate SB1 groups excluded from HO3D-MV scoring (DEGENERATE_KEYPOINT_BOX_REFERENCE_VIEW); 2,706 constructed / 2,702 scored.
- HanCo test = full UST-Hand protocol list (20,198 frames, sequences 1200-1517, 8 views, root-translation protocol).
