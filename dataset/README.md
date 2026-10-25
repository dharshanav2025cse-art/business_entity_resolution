# Business Entity Resolution – Synthetic Dataset

A synthetic dataset for building and evaluating a record-linkage / entity-resolution pipeline across three business-record sources. Every business name and address is randomly generated from word lists. None of it comes from a real company database.

Full statistics and the results of the quality checks are in `dataset_summary.txt`.

## Folder structure

```
dataset/
├── train/
│   ├── train_source1.tsv        5,000 reference businesses
│   ├── train_source2.tsv        8,000 noisy records
│   ├── train_source3.tsv        8,000 noisy records
│   └── train_ground_truth.tsv   5,000 rows (one per Source 1 entity)
├── test/
│   ├── test_source1.tsv         2,000 reference businesses
│   ├── test_source2.tsv         3,000 noisy records
│   └── test_source3.tsv         3,000 noisy records   (no ground truth)
├── README.md
└── dataset_summary.txt
```

All files are UTF-8 and **tab-separated** (written with `df.to_csv(file, sep="\t", index=False)`). Addresses can contain commas. Missing values are empty strings. Read the files with:

```python
pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
```

## Source structure

Every source file has the same four columns. These are the **only fields available to the ML model**:

| column | description |
|---|---|
| `entity_id` | `S1-00001…`, `S2-00001…`, `S3-00001…`. Unique within a source. The IDs are assigned after shuffling, so their order tells you nothing about matches. |
| `business_name` | business name, possibly with noise |
| `business_address` | free-text address, possibly with noise |
| `country` | country as a plain string (see below) |

- **Source 1 (reference):** clean, full legal names and fully formatted addresses. It includes 320 train / 130 test "look-alike" reference entities: distinct businesses deliberately similar to other Source 1 businesses.
- **Source 2 ("legacy CRM / registry export" style):** legal-suffix normalisation, abbreviations, ALL-CAPS, city aliases (Bombay, Madras…), dropped state, full US state names, ZIP+4, some typos. Almost no reordering. Country is sometimes a code (`IN`, `US`, `FR`, `IND`, `U.S.`).
- **Source 3 ("web-scraped directory" style):** dotted/spaced initials, word-order swaps, dropped and extra words ("Branch", "- Pune Office", "(India)"), transliteration, landmarks ("Near SBI ATM"), `No.`/`#` house-number prefixes, reordered addresses, lowercase, stripped French accents. Country is sometimes a long form or different case (`Republic of India`, `United States`, `FRANCE`).

## Countries

India ≈ 50%, USA ≈ 29%, France ≈ 20%, missing ≈ 2%. Country is an ordinary categorical string. Each country has its own address format:

- India: house/plot number, building and floor, street, locality, city, state, 6-digit PIN
- USA: number, direction, street and type, suite/floor, city, state, ZIP
- France: complement (bâtiment/étage), number (+bis/ter), street type and name, code postal and city

**Test** adds cities that never appear in train (Lucknow, Mysuru, Vadodara, San Diego, Charlotte, Columbus, Grenoble, Dijon, Angers), so you can check how well the model generalises.

## Business categories (20)

Technology, Software, IT Services, Manufacturing, Retail, Grocery, Restaurants, Hotels, Education, Consulting, Finance, Logistics, Healthcare, Construction, Automobile, Electronics, Clothing, Furniture, Food Processing, Professional Services.

Category is **not** a column. It only shapes the vocabulary of each name (e.g. "…Infotech", "Brasserie …", "…Motors").

## Noise types

Each noisy record gets a noise intensity of light, medium or heavy (30/50/20% in train). Each noise operation then fires independently, with a probability specific to the source, so no two records are noisy in exactly the same way.

**Names**
- legal-suffix variants (Private Limited → Pvt Ltd / Pvt. Ltd. / (P) Ltd / dropped; Inc ↔ Incorporated; Corporation → Corp; LLC → L.L.C.; SARL → S.A.R.L.; SAS…)
- occasional wrong-but-plausible suffix (Ltd vs Pvt Ltd, Inc vs LLC)
- abbreviations (Technologies → Tech, International → Intl, Engineering → Engg, Industries → Inds…)
- punctuation (A.B.C, A B C, hyphen joins, a comma before the suffix, trailing dots)
- word-order swaps; dropped generic words; extra words (Branch, Office, Unit, Division, country, "- City Branch")
- character typos (deletion, transposition, keyboard substitution, doubling)
- Indian transliteration (Sri/Shri/Shree, Lakshmi/Laxmi, Venkateswara/Venkateshwara…)
- `&` ↔ `and`; French accents stripped; upper/lower case

**Addresses**
- abbreviations (Road → Rd, Street → St, Avenue → Ave, Boulevard → Blvd/Bd, Building → Bldg, Suite → Ste/#, Rue → R.)
- Indian street aliases (Mahatma Gandhi Road → MG Road / M G Road / M.G. Road, Anna Salai → Mount Road…)
- city aliases (Bengaluru ↔ Bangalore, Mumbai ↔ Bombay, Kolkata ↔ Calcutta, Chennai ↔ Madras, Kochi ↔ Cochin, New York → NYC…)
- missing components (postal code, state, city, locality, building, floor/suite, house number)
- reordering ("Karnataka, Bangalore, MG Road, 12")
- landmarks ("Opposite Central Mall", "Beside City Hospital")
- `No.`/`#`/`Door No` prefixes
- transliteration of localities (Koramangala → Kormangala, Malleshwaram → Malleswaram)
- French arrondissement form ("Lyon 2e"), CEDEX, US ZIP+4, full state names, typos

**Test-only shifts:** more typos, including phonetic ones (ph → f, c → k); more `&`/`and` swaps and reordering; spaced or dashed PINs ("560 038", "Bengaluru - 560038"); more landmarks; and a heavier intensity mix (25/45–50/25–30%).

**Hard positives:** about 5% of true matches (330 in train) list the business at a *different* address in the same city or on the same street, as happens with a relocated business or a second office.

## Missing-value percentages (train)

| | names | blank address | incomplete address* | country |
|---|---|---|---|---|
| Source 1 | 0.0% | 0.3% | 1.3% | 1.1% |
| Source 2 | 3.5% | 2.3% | 7.1% | 2.2% |
| Source 3 | 3.2% | 2.2% | 9.0% | 2.1% |
| All train | 2.6% | 1.8% | 6.4% | 1.9% |

\*Incomplete means the address is blank, truncated to a single component, or missing 2+ of street number / city / postal code. Dropping a single component (e.g. only the postal code) is treated as ordinary noise and is far more common. Source 1 names are never missing because it is the reference source.

## Match distribution (train Source 1, 5,000 entities)

- At least one Source 2 match: 54.3% (target 55%). At least one Source 3 match: 50.4% (target 50%)
- Zero matches in both sources: 22.3%
- Exactly one match: 37.7%
- Multiple matches: 40.0%
- Matches in both Source 2 and Source 3: 27.0%
- Source 2 match count 0/1/2/3: 2283 / 1921 / 595 / 201
- Source 3 match count 0/1/2/3: 2479 / 1971 / 480 / 70
- Total true pairs: 6,855

Source 2 and Source 3 also contain businesses that are not in Source 1:
- ≈ 3,000 / 3,560 unrelated records
- 1,300 hard negatives in each source
- 312 non-S1 businesses duplicated inside Source 2
- 355 non-S1 businesses present in both S2 and S3

So "same business in S2 and S3" does not imply "matches Source 1".

## Hard-negative strategy

A hard negative is a **genuinely different business** (it has its own hidden identity) built from a real Source 1 business. There are 2,920 in train: 320 inside Source 1, 1,300 in S2 and 1,300 in S3. There are 1,130 in test.

Name transformations:
- `add_word`: Brasserie Dubois SAS → Brasserie Dubois *International* SAS; Global Foods → *New* Global Foods
- `core_swap`: ABC Technologies → ABC *Technology Labs*; Global Foods → Global *Food Products*
- `brand_change`: one brand token or initial changed (KMR → KMT)
- `sister_concern`: same brand in a different line of business (BEG Systems vs BEG Autocare)
- `chain`: **identical name**, different outlet/address (mostly shop-type categories)

About half of them keep the same legal suffix as the original.

Address relation to the original:
- same street with a nearby number (41%)
- same locality, different street (27%)
- same building, different floor or suite (17%)
- another city (15%, mostly chains)

Hard negatives then pass through the same noise pipeline as true matches. Mean name similarity (difflib ratio):

- true matches vs their S1 record: 0.91
- hard negatives vs their look-alike S1 record: 0.74
- random pairs: 0.27

As a baseline, exact name matching finds only 39% of true pairs, at 91% precision. The precision losses come from chains and look-alikes.

## Random seed

The master seed is **42**. It derives the train and test seeds. Running the generator twice gives byte-identical files (checked with md5).

## How ground truth was generated

During generation every business gets a hidden `canonical_entity_id`. Every noisy record inherits the canonical ID of the business it was rendered from. `train_ground_truth.tsv` lists, for each Source 1 `entity_id`, every S2/S3 `entity_id` that has the **same canonical ID**. It is sorted with S2 IDs first, then S3, and is empty if there are none. No similarity rule is involved, so ground truth is exact by construction. Hard negatives always have their own canonical ID and never appear in ground truth. This is checked automatically.

## Hidden fields (used only during generation – NOT in any file here)

`canonical_entity_id`, category, noise level and the noise operations applied, missing-value flags, record kind (match / hard_negative / unrelated / look-alike), hard-negative type and address relation, true country, and the look-alike group. These exist only in the generator's private `_hidden/` folder, together with the withheld test answer key. Check 16 confirms that no canonical ID column or value appears in any public TSV.

## Available to the ML model

Only `entity_id`, `business_name`, `business_address`, `country` from the six source files. For training, `train_ground_truth.tsv` can also be used.
