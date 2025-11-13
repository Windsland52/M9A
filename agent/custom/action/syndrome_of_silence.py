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
        if type in ["购物契机", "遭遇", "途中余兴", "冲突", "险象环生", "恶战"]:
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
        while retry_times < 10:
            if self.exec_action(context.clone(), action):
                return True
            # 执行中断检测
            for interrupt in interrupts:
                img = context.tasker.controller.post_screencap().wait().get()
                if self.exec_action(context.clone(), interrupt):
                    break
            time.sleep(1)
            retry_times += 1
        return False

    def exec_action(self, context: Context, action: dict | list | str) -> bool:
        # 如果是字符串,说明是 interrupt 节点，识别后执行
        if isinstance(action, str):
            if context.run_recognition(action, context.tasker.controller.cached_image):
                logger.debug(f"执行中断节点: {action}")
                context.run_task(action)
                return True
        elif isinstance(action, list):
            # 对于列表，依次执行，任意一个成功即返回成功
            for act in action:
                if self.exec_action(context, act):
                    return True
        elif isinstance(action, dict):
            # 对于单个动作，执行并检查结果
            action_type = action.get("type")
            if action_type == "RunNode":
                name = action.get("name", "")
                img = context.tasker.controller.post_screencap().wait().get()
                reco_detail = context.run_recognition(name, img)
                if (
                    reco_detail
                    and reco_detail.best_result
                    or reco_detail
                    and reco_detail.algorithm == "DirectHit"
                ):
                    logger.debug(f"执行节点: {name}")
                    context.run_task(entry=name)
                    return True
            elif action_type == "SelectOption":
                method = action.get("method")
                if method == "OCR":
                    expected_all: list[str] | str = action.get("expected", "")
                    order_by: str = action.get("order_by", "Vertical")
                    index: int = action.get("index", 0)
                    origin_node = context.get_node_data("SOSSelectOption_OCR")

                    if not origin_node:
                        logger.error("未找到原始节点 SOSSelectOption_OCR")
                        return False

                    # 将 expected 统一转为列表
                    expected_list = (
                        expected_all
                        if isinstance(expected_all, list)
                        else [expected_all]
                    )

                    pp_override = {"SOSSelectOption": {"interrupt": []}}

                    # 为每个 expected 创建独立节点
                    for i, expected in enumerate(expected_list):
                        node_name = f"SOSSelectOption_OCR_{i}"
                        # 基于 origin_node 创建新节点
                        new_node = origin_node.copy()
                        if "recognition" not in new_node:
                            new_node["recognition"] = {}
                        if "param" not in new_node["recognition"]:
                            new_node["recognition"]["param"] = {}

                        # 更新参数
                        new_node["recognition"]["param"]["expected"] = expected
                        new_node["recognition"]["param"]["order_by"] = order_by
                        new_node["recognition"]["param"]["index"] = index

                        # 添加到 pipeline_override
                        pp_override[node_name] = new_node
                        pp_override["SOSSelectOption"]["interrupt"].append(node_name)

                    context.run_task("SOSSelectOption", pipeline_override=pp_override)
                elif method == "HSV":
                    order_by: str = action.get("order_by", "Vertical")
                    index: int = action.get("index", 0)
                    pp_override = {
                        "SOSSelectOption": {"interrupt": ["SOSSelectOption_HSV"]},
                        "SOSSelectOption_HSV": {
                            "order_by": order_by,
                            "index": index,
                        },
                    }
                    context.run_task("SOSSelectOption", pipeline_override=pp_override)
                else:
                    logger.error(f"未知的选项选择方法: {method}")
                    return False
                return True
        return False
