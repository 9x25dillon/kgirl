from tauls_model import TAULSEvaluator


def test_tauls_evaluator_basic():
    ev = TAULSEvaluator()
    res = ev.score('Hello world! This is a short test.')
    assert 'entropy' in res and 'stability' in res and 'length' in res
    assert 0.0 <= res['entropy'] <= 1.0
    assert 0.0 <= res['stability'] <= 1.0
