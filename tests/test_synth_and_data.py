import random

from abstain.checker import choose_threshold
from abstain.data import Task, parent_doc, short_collection, turn_bucket
from abstain.idk import IDK_RESPONSE, is_idk
from abstain.retrieval import rrf_fuse
from abstain.synth import checker_examples, insert_gold, remove_evidence


def test_parent_doc():
    assert parent_doc("822086267_7384-8758-0-1374", "clapnq") == "822086267"
    assert parent_doc("10171-0-2129", "fiqa") == "10171"
    assert parent_doc("ibmcld_00474-7885-8455", "cloud") == "ibmcld_00474"
    assert parent_doc("c8db6e06ff46669e-50302-52227", "govt") == "c8db6e06ff46669e"
    assert short_collection("mt-rag-ibmcloud-elser-512-100-20240502") == "cloud"


def test_remove_evidence_drops_whole_document():
    ranked = [{"document_id": i, "text": i} for i in ["10-0-5", "10-6-9", "11-0-5", "12-0-5"]]
    out = remove_evidence(ranked, {"10"}, "fiqa", k=5)
    assert [p["document_id"] for p in out] == ["11-0-5", "12-0-5"]


def test_insert_gold_keeps_k_and_gold():
    ranked = [{"document_id": f"{i}-0-1", "text": ""} for i in range(10)]
    gold = [{"document_id": "99-0-1", "text": "g"}]
    out = insert_gold(ranked, gold, 5, random.Random(0))
    assert len(out) == 5 and "99-0-1" in [p["document_id"] for p in out]


def test_checker_examples_balanced_sources():
    t_ans = Task("c<::>2", "c", 2, "fiqa", [], "q", "ANSWERABLE", contexts=[{"document_id": "1-0-5", "text": "gold"}])
    t_un = Task("c<::>3", "c", 3, "fiqa", [], "q2", "UNANSWERABLE")
    t_conv = Task("c<::>4", "c", 4, "fiqa", [], "hi", "CONVERSATIONAL")
    corpus = [{"document_id": f"{i}-0-5", "text": f"p{i}"} for i in range(1, 12)]

    def retrieve(task, q, exclude):
        return [p for p in corpus if not exclude or parent_doc(p["document_id"], "fiqa") not in exclude]

    exs = checker_examples([t_ans, t_un, t_conv], retrieve, lambda t: t.question, k=3)
    kinds = sorted((e["label"], e["kind"]) for e in exs)
    assert kinds == [(0, "real"), (1, "real"), (1, "synthetic")]
    syn = [e for e in exs if e["kind"] == "synthetic"][0]
    assert all(not p["document_id"].startswith("1-") for p in syn["passages"])


def test_threshold_respects_budget():
    probs = [0.9, 0.8, 0.7, 0.6, 0.2, 0.1, 0.65]
    labels = [1, 1, 0, 1, 0, 0, 0]
    th = choose_threshold(probs, labels, max_false_abstention=0.0)
    assert th["false_abstention_rate"] == 0 and th["threshold"] > 0.7


def test_rrf():
    assert rrf_fuse([["a", "b", "c"], ["b", "a", "d"]])[:2] in (["a", "b"], ["b", "a"])
    assert rrf_fuse([["a", "b"], ["b", "c"]])[0] == "b"


def test_idk():
    assert is_idk(IDK_RESPONSE)
    assert is_idk("The passages do not contain information about that.")
    assert not is_idk("The Cardinals play at State Farm Stadium in Glendale.")


def test_bucket():
    assert turn_bucket(7, [[1, 2], [3, 5], [6, 99]], ["e", "m", "l"]) == "l"


def test_passage_examples_labels():
    from abstain.checker import passage_examples
    exs = [{"task_id": "a", "query": "q", "label": 0, "passages": [{"document_id": "g", "text": "x"}, {"document_id": "n", "text": "y"}]},
           {"task_id": "b", "query": "q", "label": 1, "passages": [{"document_id": "g", "text": "x"}, {"document_id": "n", "text": "y"}]}]
    out = passage_examples(exs, {"a": {"g"}})
    assert [e["label"] for e in out] == [0, 1, 1]        # gold of answerable -> 0, non-gold skipped, unanswerable -> 1


def test_reranker_score_checker():
    from abstain.checker import RerankerScoreChecker
    assert abs(RerankerScoreChecker.p_unanswerable([0.1, 0.9]) - 0.1) < 1e-9     # probabilities used as is
    assert RerankerScoreChecker.p_unanswerable([-5.0, 3.0]) < 0.1                # logits squashed first
    assert RerankerScoreChecker.p_unanswerable([]) == 1.0
