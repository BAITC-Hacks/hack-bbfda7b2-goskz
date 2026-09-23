import json
import unittest

from aml_agent import AMLAnalystAgent


class Item:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeResponses:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.responses = FakeResponses(responses)


class RecordingService:
    def __init__(self):
        self.executed = []

    def __getattr__(self, name):
        def handler(**arguments):
            self.executed.append((name, arguments))
            gid = arguments.get("gid", arguments.get("source_gid", 3))
            if gid == 999:
                return {"gid": 999, "found": False, "error": "gid not found"}
            if name in {"trace_downstream", "trace_upstream", "get_subgraph_for_visualization"}:
                return {"found": True, "start_gid": gid, "nodes": [{"gid": gid}, {"gid": 4}],
                        "edges": [{"source": gid, "target": 4, "amount": 50}]}
            return {"gid": gid, "found": True, "total_received": 400, "priority_score": 0.91}
        return handler


def call(name, arguments, call_id="call_1"):
    return Item(type="function_call", name=name, arguments=json.dumps(arguments), call_id=call_id)


def final(text):
    return Item(output=[], output_text=text)


class AMLAnalystAgentTests(unittest.TestCase):
    def make_agent(self, outputs):
        self.service = RecordingService()
        self.client = FakeClient(outputs)
        return AMLAnalystAgent(self.service, self.client, model="test-model")

    def test_general_question_uses_get_node(self):
        agent = self.make_agent([Item(output=[call("get_node", {"gid": 3})]), final("GID 3 has received 400.")])
        response = agent.ask("Tell me about gid 3.")
        self.assertEqual(self.service.executed[0][0], "get_node")
        self.assertIn("400", response.answer)

    def test_role_and_priority_tools_are_selected(self):
        agent = self.make_agent([Item(output=[call("get_role_explanation", {"gid": 3})]),
                                 Item(output=[call("get_priority_explanation", {"gid": 3}, "call_2")]),
                                 final("The role and priority use deterministic evidence.")])
        agent.ask("Why is gid 3 a consolidator and high priority?")
        self.assertEqual([name for name, _ in self.service.executed], ["get_role_explanation", "get_priority_explanation"])

    def test_direct_connections_and_trace_are_distinct(self):
        direct = self.make_agent([Item(output=[call("get_node_connections", {"gid": 3, "direction": "outgoing"})]), final("Direct recipients returned.")])
        direct.ask("Where does gid 3 send money?")
        self.assertEqual(self.service.executed[0][0], "get_node_connections")
        traced = self.make_agent([Item(output=[call("trace_downstream", {"gid": 3, "hops": 3})]), final("Downstream trace returned.")])
        response = traced.ask("Trace downstream from gid 3.")
        self.assertEqual(self.service.executed[0][0], "trace_downstream")
        self.assertEqual(response.visualization["focus_node"], 3)

    def test_common_recipient_and_compare(self):
        agent = self.make_agent([Item(output=[call("find_common_recipient", {"gids": [1, 2, 3]})]), final("One common recipient was found.")])
        agent.ask("Who collects money from gids 1, 2 and 3?")
        self.assertEqual(self.service.executed[0][0], "find_common_recipient")
        agent = self.make_agent([Item(output=[call("compare_nodes", {"gids": [3, 7]})]), final("Comparable metrics returned.")])
        agent.ask("Compare gids 3 and 7.")
        self.assertEqual(self.service.executed[0][0], "compare_nodes")

    def test_unknown_gid_does_not_create_metrics(self):
        agent = self.make_agent([Item(output=[call("get_node", {"gid": 999})]), final("I could not find gid 999 in the loaded transaction network.")])
        response = agent.ask("Tell me about gid 999.")
        self.assertIn("could not find", response.answer)
        self.assertEqual(response.referenced_gids, [999])

    def test_multi_step_and_history_are_forwarded(self):
        agent = self.make_agent([Item(output=[call("get_priority_explanation", {"gid": 3})]),
                                 Item(output=[call("get_node_connections", {"gid": 3, "direction": "outgoing"}, "call_2")]),
                                 final("Priority and outgoing flows were reviewed.")])
        response = agent.ask("Why is it high priority and where does it send money?", [{"role": "user", "content": "Tell me about gid 3."}])
        self.assertEqual(len(response.tool_calls), 2)
        self.assertEqual(self.client.responses.requests[0]["input"][0]["content"], "Tell me about gid 3.")


if __name__ == "__main__":
    unittest.main()
