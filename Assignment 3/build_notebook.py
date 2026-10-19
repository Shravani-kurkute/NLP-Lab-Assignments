#!/usr/bin/env python3
"""Create and execute POS_Tagging_Assignment_03.ipynb.

The notebook is built programmatically, executed with nbclient using the
project directory as its working directory, and saved with all outputs.
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

PROJECT_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_DIR / "POS_Tagging_Assignment_03.ipynb"
EXECUTED_PATH = PROJECT_DIR / "outputs" / "POS_Tagging_Assignment_03_executed.ipynb"


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata.update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
            "title": "NLP Laboratory Assignment 3 - HMM POS Tagging",
            "authors": [{"name": "Shravani Kurkute"}],
        }
    )

    cells = []
    cells.append(
        nbf.v4.new_markdown_cell(
            "# NLP Laboratory Assignment 3\n"
            "## Survey and Implementation of Part-of-Speech (POS) Tagging\n\n"
            "**Student:** Shravani Kurkute  \n"
            "**Technique:** Statistical Hidden Markov Model with Viterbi decoding  \n"
            "**Corpus:** NLTK Penn Treebank, mapped to the NLTK Universal tagset  \n"
            "**Configuration:** 80:20 sentence-level split, seed 42, add-k transition smoothing "
            "(k = 0.001), digit/suffix OOV model\n\n"
            "This notebook reproduces the same experiment as `pos_tagging_hmm.py`. Every cell is "
            "executable and the saved output corresponds to an actual run."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "# 1. Imports and environment setup\n"
            "import sys, math, json\n"
            "from collections import Counter\n"
            "from pathlib import Path\n"
            "\n"
            "PROJECT_DIR = Path.cwd()\n"
            "if str(PROJECT_DIR) not in sys.path:\n"
            "    sys.path.insert(0, str(PROJECT_DIR))\n"
            "\n"
            "import nltk, numpy as np, pandas as pd, sklearn, matplotlib\n"
            "from IPython.display import display, Image\n"
            "\n"
            "import pos_tagging_hmm as pos\n"
            "\n"
            "pos.set_random_seeds(pos.SEED)\n"
            "pos.ensure_directories()\n"
            "print('Project directory :', PROJECT_DIR)\n"
            "print('Python            :', sys.version.split()[0])\n"
            "print('NLTK              :', nltk.__version__)\n"
            "print('NumPy             :', np.__version__)\n"
            "print('pandas            :', pd.__version__)\n"
            "print('scikit-learn      :', sklearn.__version__)\n"
            "print('Matplotlib        :', matplotlib.__version__)\n"
            "print('Seed              :', pos.SEED)\n"
            "print('Transition k      :', pos.TRANSITION_K)"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "# 2. Dataset download and loading\n"
            "resources = pos.ensure_nltk_resources(allow_download=True, verbose=True)\n"
            "raw_sentences, load_stats = pos.load_treebank_sentences(remove_traces=True)\n"
            "display(pd.DataFrame(load_stats.items(), columns=['Statistic', 'Value']))\n"
            "print('First tagged sentence:')\n"
            "print(raw_sentences[0])"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "# 3. Dataset inspection\n"
            "ptb_tag_counts = Counter(tag for sent in raw_sentences for _, tag in sent)\n"
            "print('Sentences:', len(raw_sentences))\n"
            "print('Tokens   :', sum(len(s) for s in raw_sentences))\n"
            "print('PTB tags :', len(ptb_tag_counts))\n"
            "display(pd.DataFrame(ptb_tag_counts.most_common(15), columns=['PTB tag', 'Count']))\n"
            "print('Example with token/PTB-tag format:')\n"
            "print(' '.join(f'{w}/{t}' for w, t in raw_sentences[1]))"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "# 4. Tagset mapping and train/test split\n"
            "universal_sentences, tag_mapping = pos.map_ptb_to_universal(raw_sentences)\n"
            "universal_sentences, duplicates_removed = pos.deduplicate_sentences(universal_sentences)\n"
            "train_sentences, test_sentences, split_meta = pos.prepare_dataset(\n"
            "    universal_sentences, test_size=pos.TEST_SIZE, seed=pos.SEED\n"
            ")\n"
            "pos.write_sentence_file(PROJECT_DIR / 'train_sentences.txt', train_sentences)\n"
            "pos.write_sentence_file(PROJECT_DIR / 'test_sentences.txt', test_sentences)\n"
            "print('Exact duplicate sentences removed:', duplicates_removed)\n"
            "display(pd.DataFrame(sorted(tag_mapping.items()), columns=['PTB tag', 'Universal tag']))\n"
            "print('Split:', split_meta)\n"
            "train_keys = {tuple(s) for s in train_sentences}\n"
            "test_keys = {tuple(s) for s in test_sentences}\n"
            "print('Sentence overlap:', len(train_keys & test_keys))\n"
            "print('First training sentence:')\n"
            "print(' '.join(w for w, _ in train_sentences[0]))\n"
            "print('First test sentence:')\n"
            "print(' '.join(w for w, _ in test_sentences[0]))"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "# 5. HMM training\n"
            "hmm = pos.HMMPosTagger(\n"
            "    transition_k=pos.TRANSITION_K,\n"
            "    rare_word_threshold=pos.RARE_WORD_THRESHOLD,\n"
            "    max_suffix_len=pos.MAX_SUFFIX_LEN,\n"
            "    oov_alpha=pos.OOV_ALPHA,\n"
            ").train(train_sentences)\n"
            "baseline = pos.MostFrequentTagBaseline().train(train_sentences)\n"
            "\n"
            "print('Universal tags      :', hmm.tags)\n"
            "print('Training vocabulary :', f'{len(hmm.vocab):,}')\n"
            "print('Emission pairs      :', f'{sum(len(c) for c in hmm.emission_counts.values()):,}')\n"
            "print('Rare-word occurrences for OOV model:', f'{hmm.rare_token_occurrences:,}')\n"
            "print('Numeric rare tokens :', f'{hmm.numeric_rare_occurrences:,}')\n"
            "\n"
            "# Show add-k smoothed transition probabilities from a few frequent tags.\n"
            "sample_tags = ['DET', 'NOUN', 'VERB', 'ADP', '.']\n"
            "transition_df = pd.DataFrame(\n"
            "    {f'{prev} ->': [math.exp(hmm.log_transition[prev][curr]) for curr in sample_tags]\n"
            "     for prev in [hmm.START, 'DET', 'NOUN', 'VERB']},\n"
            "    index=sample_tags,\n"
            ").T\n"
            "display(transition_df.round(4))"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "### 6. Viterbi decoding: explanation\n"
            "\n"
            "For an HMM, Viterbi computes\n"
            "\n"
            "$$\\delta_i(j) = \\max_k \\left[\\delta_{i-1}(k) + \\log P(t_j \\mid t_k)\\right] + \\log P(w_i \\mid t_j)$$\n"
            "\n"
            "and stores $\\psi_i(j)=\\arg\\max_k[\\delta_{i-1}(k)+\\log P(t_j\\mid t_k)]$.\n"
            "After the final token, $\\log P(END\\mid t_j)$ is added, the best final tag is selected, "
            "and backpointers recover the path. The demonstration below uses the actual trained model "
            "and displays the score matrix and chosen predecessor for each position."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "# 6. Viterbi decoding demonstration\n"
            "tokens = ['The', 'committee', 'approved', 'the', 'budget', '.']\n"
            "tags, log_prob, details = hmm.viterbi_decode(tokens, return_details=True)\n"
            "print('Tokens     :', tokens)\n"
            "print('Predicted  :', tags)\n"
            "print('Log score  :', round(log_prob, 6))\n"
            "print('Tagged     :', ' '.join(f'{w}/{t}' for w, t in zip(tokens, tags)))\n"
            "\n"
            "score_df = pd.DataFrame(details['score_matrix'], index=tokens, columns=details['tags'])\n"
            "display(score_df.round(3))\n"
            "\n"
            "backpointer_df = pd.DataFrame(\n"
            "    [[('START' if index < 0 else details['tags'][index]) for index in row]\n"
            "     for row in details['backpointers']],\n"
            "    index=tokens,\n"
            "    columns=details['tags'],\n"
            ")\n"
            "print('Chosen predecessor for every (position, tag):')\n"
            "display(backpointer_df)"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "# 7. Baseline comparison\n"
            "test_words, test_tags = pos.flatten_tagged(test_sentences)\n"
            "train_words, train_tags = pos.flatten_tagged(train_sentences)\n"
            "train_vocab = set(train_words)\n"
            "\n"
            "hmm_predictions = [hmm.tag([w for w, _ in sent]) for sent in test_sentences]\n"
            "baseline_predictions = [baseline.tag([w for w, _ in sent]) for sent in test_sentences]\n"
            "y_true = pos.flatten_predictions([[t for _, t in sent] for sent in test_sentences])\n"
            "y_hmm = pos.flatten_predictions(hmm_predictions)\n"
            "y_base = pos.flatten_predictions(baseline_predictions)\n"
            "\n"
            "hmm_metrics, hmm_per_tag, cm = pos.evaluate_tags(y_true, y_hmm, hmm.tags, test_words, train_vocab)\n"
            "base_metrics, base_per_tag, _ = pos.evaluate_tags(y_true, y_base, hmm.tags, test_words, train_vocab)\n"
            "\n"
            "comparison_df = pd.DataFrame({\n"
            "    'Metric': ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1',\n"
            "               'Weighted Precision', 'Weighted Recall', 'Weighted F1',\n"
            "               'OOV Accuracy', 'Known Accuracy'],\n"
            "    'Baseline': [base_metrics['accuracy'], base_metrics['macro_precision'], base_metrics['macro_recall'],\n"
            "                 base_metrics['macro_f1'], base_metrics['weighted_precision'], base_metrics['weighted_recall'],\n"
            "                 base_metrics['weighted_f1'], base_metrics['oov_accuracy'], base_metrics['known_accuracy']],\n"
            "    'HMM': [hmm_metrics['accuracy'], hmm_metrics['macro_precision'], hmm_metrics['macro_recall'],\n"
            "            hmm_metrics['macro_f1'], hmm_metrics['weighted_precision'], hmm_metrics['weighted_recall'],\n"
            "            hmm_metrics['weighted_f1'], hmm_metrics['oov_accuracy'], hmm_metrics['known_accuracy']],\n"
            "})\n"
            "comparison_df['Difference'] = comparison_df['HMM'] - comparison_df['Baseline']\n"
            "display(comparison_df.round(6))"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "# 8. Evaluation metrics and per-tag report\n"
            "display(hmm_per_tag.round(4))\n"
            "print('HMM classification report:')\n"
            "print(hmm_metrics['classification_report'])\n"
            "print('Test tokens:', hmm_metrics['support'])\n"
            "print('OOV tokens :', hmm_metrics['oov_count'], f\"({hmm_metrics['oov_rate']*100:.2f}%)\")\n"
            "print('HMM OOV accuracy     :', round(hmm_metrics['oov_accuracy'], 6))\n"
            "print('HMM known accuracy   :', round(hmm_metrics['known_accuracy'], 6))"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "# 9. Custom sentence predictions (previously unseen)\n"
            "custom_sentences = pos.load_custom_sentences(PROJECT_DIR / 'custom_test_sentences.txt')\n"
            "custom_records, custom_rows = pos.run_custom_tests(custom_sentences, hmm, baseline)\n"
            "for record in custom_records:\n"
            "    print(f\"{record['sentence_id']:02d}. {record['sentence']}\")\n"
            "    print('    HMM     :', record['hmm_tagged'])\n"
            "    print('    Baseline:', record['baseline_tagged'])\n"
            "    print()"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "# 10. Error analysis\n"
            "error_rows, top_errors = pos.build_error_rows(\n"
            "    test_sentences, hmm_predictions, baseline_predictions, train_vocab\n"
            ")\n"
            "print('Total HMM token errors:', len(error_rows))\n"
            "top_error_df = pd.DataFrame(top_errors[:10])\n"
            "display(top_error_df)\n"
            "print('First five error contexts:')\n"
            "for row in error_rows[:5]:\n"
            "    print('-', row['context'])"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "# 11. Visualizations (regenerated from the same configuration)\n"
            "bundle = pos.run_pipeline(write_files=False, verbose=False)\n"
            "result = bundle['result']\n"
            "charts = pos.generate_visualizations(result, bundle['y_true'], bundle['y_pred'])\n"
            "for chart in charts:\n"
            "    print(chart)\n"
            "    display(Image(filename=str(PROJECT_DIR / chart), width=820))"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "# 12. Saving results and validating outputs\n"
            "created = pos.save_core_outputs(bundle)\n"
            "print('Files written:')\n"
            "for item in created:\n"
            "    print(' -', item)\n"
            "required = [\n"
            "    'results.json', 'evaluation_results.csv', 'sample_tagged_output.csv',\n"
            "    'error_analysis.csv', 'train_sentences.txt', 'test_sentences.txt',\n"
            "    'custom_test_sentences.txt', 'visualizations/confusion_matrix.png',\n"
            "    'visualizations/per_tag_scores.png', 'visualizations/model_comparison.png',\n"
            "    'visualizations/oov_analysis.png', 'visualizations/tag_distribution.png',\n"
            "]\n"
            "for relative in required:\n"
            "    assert (PROJECT_DIR / relative).exists(), relative\n"
            "print('All required core artefacts exist.')\n"
            "display(pd.DataFrame(result['summary']))"
        )
    )

    nb.cells = cells
    return nb


def main() -> int:
    nb = build_notebook()
    NOTEBOOK_PATH.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"Notebook source written: {NOTEBOOK_PATH}")

    client = NotebookClient(
        nb,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_DIR)}},
        allow_errors=False,
    )
    client.execute()

    nbf.write(nb, str(NOTEBOOK_PATH))
    EXECUTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(EXECUTED_PATH))

    code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    executed = sum(1 for cell in code_cells if cell.get("execution_count") is not None)
    errors = sum(
        1
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    )
    print(f"Notebook executed: {executed}/{len(code_cells)} code cells, errors={errors}")
    print(f"Saved executed notebook: {NOTEBOOK_PATH}")
    print(f"Backup copy: {EXECUTED_PATH}")
    if errors:
        raise RuntimeError("Notebook execution produced error outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
