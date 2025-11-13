import time
import json
from typing import Union, cast

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.define import NeuralNetworkDetectResult, OCRResult, ColorMatchResult

from utils import logger


@AgentServer.custom_action("SOSSelectNode")
class SOSSelectNode(CustomAction):
    """
    节点选择
    """

    type, event = "", ""

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        reco_detail = cast(NeuralNetworkDetectResult, argv.reco_detail.best_result)

        with open("resource/data/sos/nodes.json", encoding="utf-8") as f:
            nodes = json.load(f)

        type = nodes["types"][reco_detail.cls_index]
        SOSSelectNode.type = type
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

        event_name_roi = nodes[type]["event_name_roi"]

        if event_name_roi:
            # 看下当前事件名
            img = context.tasker.controller.post_screencap().wait().get()
            reco_detail = context.run_recognition(
                "SOSEventRec", img, {"roi": event_name_roi}
            )
            if reco_detail:
                ocr_result = cast(OCRResult, reco_detail.best_result)
                event = ocr_result.text
                SOSSelectNode.event = event
                logger.info(f"当前事件: {event}")
            else:  # 识别失败
                SOSSelectNode.event = ""
                return CustomAction.RunResult(success=False)
        else:
            # 没有事件名
            SOSSelectNode.event = ""
        return CustomAction.RunResult(success=True)


@AgentServer.custom_action("SOSNodeProcess")
class SOSNodeProcess(CustomAction):
    """
    节点处理
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        with open("resource/data/sos/nodes.json", encoding="utf-8") as f:
            nodes = json.load(f)

        type, event = (
            SOSSelectNode.type,
            SOSSelectNode.event,
        )

        # 无 event 的处理
        if type in ["遭遇", "途中余兴", "冲突", "险象环生", "恶战"]:
            actions: list = nodes[type]["actions"]
            interrupts: list = nodes[type].get("interrupts", [])
        else:
            # 有 event 的处理
            if event not in nodes[type]["events"]:
                logger.error(f"未适配该事件: {event}")
                context.tasker.post_stop()
                return CustomAction.RunResult(success=False)

            info: dict = nodes[type]["events"][event]
            actions: list = info["actions"]
            interrupts: list = info.get("interrupts", [])

        for action in actions:
            if not self.exec_main(context, action, interrupts):
                return CustomAction.RunResult(success=False)
        return CustomAction.RunResult(success=True)

    def exec_main(self, context: Context, action: dict | list, interrupts: list):
        retry_times = 0
        while retry_times < 3:
            if self.exec_action(context.clone(), action):
                return True
            # 执行中断检测
            for interrupt in interrupts:
                if self.exec_action(context.clone(), interrupt):
                    break
            retry_times += 1
        return False

    def exec_action(self, context: Context, action: dict | list) -> bool:
        if isinstance(action, list):
            # 对于列表，依次执行，任意一个成功即返回成功
            for act in action:
                if self.exec_action(context, act):
                    return True
        else:
            # 对于单个动作，执行并检查结果
            type = action.get("type")
            if type == "RunNode":
                name = action.get("name", "")
                logger.debug(f"执行节点: {name}")
                img = context.tasker.controller.post_screencap().wait().get()
                reco_detail = context.run_recognition(name, img)
                if reco_detail and reco_detail.best_result and reco_detail.box:
                    context.run_action(entry=name, box=reco_detail.box)
                    return True
            elif type == "SelectOption":
                method = action.get("method")
                if method == "OCR":
                    expected_all: list[str] | str = action.get("expected", "")
                    order_by: str = action.get("order_by", "Vertical")
                    index: int = action.get("index", 0)
                    for expected in (
                        expected_all
                        if isinstance(expected_all, list)
                        else [expected_all]
                    ):
                        img = context.tasker.controller.post_screencap().wait().get()
                        reco_detail = context.run_recognition(
                            "SOSSelectOption_OCR",
                            img,
                            {
                                "SOSSelectOption_OCR": {
                                    "expected": expected,
                                    "order_by": order_by,
                                    "index": index,
                                }
                            },
                        )
                        if reco_detail and reco_detail.best_result and reco_detail.box:
                            context.run_action(
                                "Click",
                                pipeline_override={
                                    "action": "Click",
                                    "target": reco_detail.box,
                                },
                            )
                            return True
                elif method == "HSV":
                    order_by: str = action.get("order_by", "Vertical")
                    index: int = action.get("index", 0)
                    img = context.tasker.controller.post_screencap().wait().get()
                    reco_detail = context.run_recognition(
                        "SOSSelectOption_HSV",
                        img,
                        {
                            "SOSSelectOption_HSV": {
                                "order_by": order_by,
                                "index": index,
                            }
                        },
                    )
                    if reco_detail and reco_detail.best_result:
                        context.run_action(
                            "Click",
                            pipeline_override={
                                "action": "Click",
                                "target": reco_detail.box,
                            },
                        )
                        return True
        return False
