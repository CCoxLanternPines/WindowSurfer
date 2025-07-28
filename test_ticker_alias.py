import importlib
import sys
import types

# stub modules before importing execution_handler
price_response = {"result": {"XDGUSD": {"c": ["0.1"]}}}
requests_stub = types.SimpleNamespace(get=lambda url: types.SimpleNamespace(json=lambda: price_response))
kraken_auth_stub = types.SimpleNamespace(load_kraken_keys=lambda: ("key", "secret"))
logger_stub = types.SimpleNamespace(addlog=lambda *a, **k: None)
resolve_symbol_stub = types.SimpleNamespace(resolve_symbol=lambda tag: {"kraken": "DOGE/USD"})

sys.modules['requests'] = requests_stub
sys.modules['systems.scripts.kraken_auth'] = kraken_auth_stub
sys.modules['systems.utils.logger'] = logger_stub
sys.modules['systems.utils.resolve_symbol'] = resolve_symbol_stub

execution_handler = importlib.import_module('systems.scripts.execution_handler')

class DummyKraken:
    def __init__(self):
        self.call_count = 0

    def __call__(self, endpoint, data, api_key, api_secret):
        if endpoint == 'Balance':
            return {'result': {'ZUSD': '1000'}}
        if endpoint == 'AddOrder':
            return {'result': {'txid': ['1']}}
        if endpoint == 'TradesHistory':
            return {'result': {'trades': {'t1': {'ordertxid': '1', 'price': '0.1', 'vol': '10', 'cost': '1', 'fee': '0.01', 'time': 1}}}}
        return {'result': {}}

execution_handler._kraken_request = DummyKraken()

def test_buy_order_alias():
    result = execution_handler.buy_order('DOGEUSD', 1.0)
    assert result['price'] == 0.1

def test_sell_order_alias():
    result = execution_handler.sell_order('DOGEUSD', 1.0)
    assert result['price'] == 0.1
