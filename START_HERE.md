# Start here

This page is for teammates who want to see the results without learning Git. You only need a web browser.

The project ranks Boston's routine restaurant inspections by how likely each one is to fail, so the city can visit the riskiest places first. The [README](README.md) has the full results table and the technical details.

## The results in four charts

How often a visit finds a failure, if inspectors make the riskiest 20% of visits first. Random order finds a failure 59% of the time, a simple rule 80%, and our model 88%.

![Hit rate by method](results/figures/hit_rate_bar.png)

What that means per year: the same 1,201 visits find about 349 more failed inspections than random order.

![Failed inspections found per year](results/figures/impact_bar.png)

What the inspector comments add on top of the inspection history: about one point. Comments on their own reach 84%, because they mostly repeat what the history already shows.

![What inspector comments add](results/figures/text_value_bar.png)

A policy question for the city. Zip code is a model input, so neighborhoods with the same fail rate get very different shares of the early visits (Dorchester 46%, Roxbury 19%).

![Share of first visits by neighborhood](results/figures/neighborhood_first_visits.png)

The other charts are in [results/figures](results/figures).

## Where things are

| Folder or file | What's in it | Open it when you want to |
|---|---|---|
| [README.md](README.md) | One-page summary: results, method, limitations | Quote a number or explain the method |
| [results/figures](results/figures) | All 10 charts as PNG images | Put a chart on a slide |
| [results/results.csv](results/results.csv) | The main results table | Check an exact number |
| [results/impact.csv](results/impact.csv) | Impact numbers, each with a line on how it was computed | Check an impact claim |
| [results/priority_list.csv](results/priority_list.csv) | Restaurants ranked by estimated risk | Read [the note](results/priority_list_note.md) first: the scores are estimates, not findings about any business |
| [notebooks](notebooks) | The analysis, step by step (01 data, 02 baselines, 03 models, 04 results, 05 fine-tuning) | See how a number was produced |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Why we made each modeling choice | Answer "why did you do it this way?" |
| [src](src) | The Python code the notebooks call | You can skip this folder |

The raw data is not in the repo on purpose. The inspector comments contain people's names and emails.

## How to use GitHub here

Click any file name to open it. GitHub shows charts as images, CSV files as tables you can search, and notebooks with their code and outputs, so you don't need to install anything.

To save one chart, open it and click the download button (the arrow icon at the top right of the file view).

To get a copy of everything, go to the repo's main page, click the green **Code** button, then **Download ZIP**.

To suggest a change, message the repo owner or open an issue (the **Issues** tab, then **New issue**). Please don't edit files directly in the browser: every edit goes straight into the shared history.

## Terms you'll see

| Term | Meaning |
|---|---|
| Hit rate | Share of visits that find a failure |
| First 20% of visits | The riskiest fifth of inspections, visited first. Our stand-in for limited inspector time |
| Random order | Our assumption for how inspections are ordered today. We could not confirm the city's actual method |
| Simple rule | Rank restaurants by how often they failed before. No model needed |
| Validation year (2023) | The year we used to choose settings |
| Test years (2024 to 2026) | Years the model never saw while we chose settings. All headline numbers come from here |
| LightGBM | The machine-learning model we chose: many small decision trees, each correcting the last |
| Embeddings | Comment text turned into numbers by a language model so the main model can use it |
| 95% range | How much a number could move by chance. Bars with overlapping ranges are not clearly different |
