import time
import json

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils import logger


@AgentServer.custom_action("SOSSelectNode")
class SOSSelectNode(CustomAction):
    """
    节点选择
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        reco_detail = argv.reco_detail.best_result

        with open("resource/data/sos/nodes.json", encoding="utf-8") as f:
            nodes = json.load(f)

        type = nodes["types"][reco_detail.cls_index]
        logger.info(f"当前进入节点类型: {type}")

        times = 0
        while times < 5:
            context.run_task(
                "Click",
                {
                    "Click": {
                        "action": "Click",
                        "target": reco_detail.box,
                        "post_wait_freezes": {
                            "time": 500,
                            "target": [846, 555, 406, 68],
                            "timeout": 5000,
                        },
                    }
                },
            )
            img = context.tasker.controller.post_screencap().wait().get()
            if context.run_task("SOSGOTO"):
                break
            times += 1

        # 看下当前事件名
        img = context.tasker.controller.post_screencap().wait().get()
        reco_detail = context.run_recognition("SOSEventRec", img)
        if reco_detail:
            event = reco_detail.best_result.text
            logger.info(f"当前事件: {event}")
            context.override_next("SOSSelectNode", nodes[type][event])
            return CustomAction.RunResult(success=True)

        # 没识别到
        context.override_next("SOSSelectNode", [])
        return CustomAction.RunResult(success=False)
