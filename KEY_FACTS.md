# Key Facts and Results

A proof of concept that proposes ERP data (production work phases and a Bill of Materials) for a 2D
technical drawing by reusing similar past drawings. Tested on ABB Drives (sheet metal) and Konecranes
(welded), about 1,800 drawings. Performance figures are decimals from 0 to 1. Dataset properties are
percentages.

## Headline results

| Task | Result | Notes |
|---|---|---|
| Work phase prediction | macro F1 0.91, exact 0.60, micro 0.93 | the strong, deployable result. Works the same on both vendors (ABB 0.93, KC 0.90). |
| Similar drawing retrieval | Success@5 0.84, P@1 0.69 | frozen DINOv2 mean pool, judged by people. The simple model beat the adapted one. |
| BOM candidate pool | recall 0.74 to 0.91 on recurring components | a shortlist, not a full BOM (see below). |
| Content extraction | title block fields and on drawing BOM read reliably | useful on its own. |

## What is fundamentally limited

- The BOM cannot be generated, only retrieved. About 79% of components are one offs that never
  repeat, so the tool surfaces a candidate pool for a human to finish. It is not a full predictor.
- BOM reuse is vendor specific. Konecranes scores 0.93, ABB scores 0.46. Report per vendor, not the
  average.
- Looking alike does not mean the same structure. A visual match means "search here first", not
  "same part".
- Text leads the prediction. Visual leads the similarity search. Combining them adds very little
  (about 0.02).

## Dataset facts

About 99.95% of drawings are linked to their ERP record. About 94.2% are top level assemblies. About
52.1% are multi page.

## Recommendation

1. Ship work phase prediction first. It is accurate, works on both vendors, and is directly useful.
2. Use the BOM as an assisted shortlist, not an autocomplete.
3. Use similar drawing retrieval to surface precedents for an engineer to adapt.
4. Keep extraction as a standalone feature.

The prototype in this repository does all four as one runnable tool.
