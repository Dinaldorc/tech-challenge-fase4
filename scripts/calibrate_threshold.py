"""Calibra o corte de similaridade abaixo do qual o sistema recusa a pergunta por falta de evidência.

Para cada pergunta de ``data/eval/threshold_questions.csv`` (dentro e fora do tema, específicas e
genéricas, com e sem filtros de metadados), registra a similaridade da melhor evidência encontrada
pela busca, com os mesmos filtros do uso real. Depois testa uma faixa de cortes:

- **recusa indevida:** pergunta do tema com melhor evidência abaixo do corte;
- **aceite indevido:** pergunta fora do tema com melhor evidência no corte ou acima dele.

As perguntas são divididas em duas metades, alternadamente dentro de cada tipo (tema, genérica,
fora, armadilha), para que todos os tipos apareçam nas duas. O corte é
escolhido na metade de calibração e conferido na de validação, que não participou da escolha.
Critério: o maior corte sem nenhuma recusa indevida na calibração. O corte é a primeira de duas
barreiras: uma pergunta fora do tema que passar por ele ainda pode ser recusada pelo LLM, mas uma
pergunta legítima recusada aqui nunca chega ao LLM. Por isso, só a recusa indevida é irreversível.
Margem de segurança: o corte final fica um passo (``--margin``, 0,005) abaixo desse valor. Sem ela, a
pergunta legítima de menor similaridade pode ficar exatamente em cima do corte, e qualquer variação
de formulação a faria ser recusada.
Para comparação, o script também mostra o corte de maior acerto, que trata os dois erros como iguais.

Uso:
    python scripts/calibrate_threshold.py                                  # base completa (chroma_db)
    python scripts/calibrate_threshold.py --persist-dir chroma_db_sample   # amostra
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.retrieval import load_retriever, parse_filters  # noqa: E402
from src.vector_store import count_documents  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, default=ROOT / "data" / "eval" / "threshold_questions.csv")
    parser.add_argument("--persist-dir", type=Path, default=ROOT / "chroma_db")
    parser.add_argument("--min", type=float, default=0.78)
    parser.add_argument("--max", type=float, default=0.90)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument("--margin", type=float, default=0.005, help="margem abaixo do maior corte sem recusa indevida")
    return parser.parse_args()


def errors(frame: pd.DataFrame, threshold: float) -> tuple[int, int]:
    false_refusals = int(((frame["best"] < threshold) & frame["in_domain"]).sum())
    false_accepts = int(((frame["best"] >= threshold) & ~frame["in_domain"]).sum())
    return false_refusals, false_accepts


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()

    questions = pd.read_csv(args.questions, keep_default_na=False)
    questions["in_domain"] = questions["in_domain"].astype(str).str.lower() == "true"
    position = questions.groupby(["in_domain", "kind"]).cumcount()
    questions["split"] = np.where(position % 2 == 0, "calibracao", "validacao")

    retriever = load_retriever(args.persist_dir, min_similarity=0.0)
    total = count_documents(retriever.vector_store)
    print(f"Índice: {args.persist_dir} ({total:,} documentos)".replace(",", "."))

    rows = []
    for question in questions.itertuples():
        result = retriever.search(question.question, k=1, filters=parse_filters(question.filters))
        top = result.evidences[0] if result.evidences else None
        rows.append({"best": result.best_similarity or 0.0, "top_text": top.text[:60] if top else "",
                     "sentiment_filter": ",".join(result.filters.sentiments or ())})
    questions = pd.concat([questions, pd.DataFrame(rows)], axis=1)

    print("\n## Similaridade da melhor evidência, da maior para a menor")
    for row in questions.sort_values("best", ascending=False).itertuples():
        label = row.kind.upper()[:5].ljust(5)
        applied = ";".join(filter(None, [row.filters, f"sentimento={row.sentiment_filter}" * bool(row.sentiment_filter)]))
        flag = f" [{applied}]" if applied else ""
        print(f"  {row.best:.3f}  {label} {row.question[:50]:<50}{flag}  -> {row.top_text}")

    print()
    for kind, values in questions.groupby("kind", sort=False)["best"]:
        print(f"{kind:<10} ({len(values):>2}): mínimo {values.min():.3f}, mediana {values.median():.3f}, "
              f"máximo {values.max():.3f}")

    calibration = questions[questions["split"] == "calibracao"]
    validation = questions[questions["split"] == "validacao"]
    thresholds = np.round(np.arange(args.min, args.max + 1e-9, args.step), 3)

    print("\n## Erros por corte (recusas indevidas / aceites indevidos)")
    print(f"  {'corte':>6}  {'calibração':>12}  {'validação':>12}")
    chosen, most_accurate, best_correct = None, None, -1
    for threshold in thresholds:
        cal = errors(calibration, threshold)
        val = errors(validation, threshold)
        print(f"  {threshold:6.3f}  {cal[0]:>5} / {cal[1]:<4}  {val[0]:>5} / {val[1]:<4}")
        if cal[0] == 0:
            chosen = threshold
        correct = len(calibration) - sum(cal)
        if correct > best_correct:  # empate mantém o menor corte
            most_accurate, best_correct = threshold, correct

    no_refusal = chosen
    chosen = round(no_refusal - args.margin, 3)
    for label, threshold in [("Maior corte sem recusa indevida na calibração", no_refusal),
                             (f"Corte escolhido (o anterior menos a margem de {args.margin:.3f})", chosen),
                             ("Para comparação, corte de maior acerto", most_accurate)]:
        print(f"\n{label}: {threshold:.3f}")
        for name, frame in [("calibração", calibration), ("validação", validation), ("todas", questions)]:
            refusals, accepts = errors(frame, threshold)
            print(f"  {name:<10}: {refusals} recusas indevidas, {accepts} aceites indevidos "
                  f"(acerto {len(frame) - refusals - accepts}/{len(frame)})")

    print(f"\n## Por tipo de pergunta, no corte escolhido ({chosen:.3f})")
    for kind, frame in questions.groupby("kind", sort=False):
        passed = int((frame["best"] >= chosen).sum())
        verb = "passam pela busca" if frame["in_domain"].iloc[0] else "passam pela busca (o LLM precisa barrar)"
        print(f"  {kind:<10}: {passed}/{len(frame)} {verb}")


if __name__ == "__main__":
    main()
