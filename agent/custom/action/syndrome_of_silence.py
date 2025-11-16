import re
import time
import json
import copy
from typing import cast
import numpy as np

from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context
from maa.define import NeuralNetworkDetectResult, OCRResult

from utils import logger


__all__ = [
    "SOSSelectNode",
    "SOSNodeProcess",
    "SOSSelectEncounterOption_OCR",
    "SOSSelectEncounterOption_HSV",
    "SOSShoppingList",
    "SOSBuyItems",
]


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
                            "timeout": 3000,
                        },
                    }
                },
            )
            img = context.tasker.controller.post_screencap().wait().get()
            if context.run_recognition("SOSGOTO", img):
                context.run_task("SOSGOTO")
                break
            times += 1

        event_name_roi = nodes[type]["event_name_roi"]

        if event_name_roi:
            # 看下当前事件名
            img = context.tasker.controller.post_screencap().wait().get()
            reco_detail = context.run_recognition(
                "SOSEventRec", img, {"SOSEventRec": {"roi": event_name_roi}}
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
        if type in ["购物契机", "遭遇", "途中余兴", "冲突", "恶战", "巧匠之手"]:
            actions: list = nodes[type]["actions"] + [
                {"type": "RunNode", "name": "FlagInSOSMain"}
            ]
            interrupts: list = nodes[type].get("interrupts", [])
        else:
            # 有 event 的处理
            if event not in nodes[type]["events"]:
                logger.error(f"未适配该事件: {event}")
                context.tasker.post_stop()
                return CustomAction.RunResult(success=False)

            info: dict = nodes[type]["events"][event]
            actions: list = info["actions"] + [
                {"type": "RunNode", "name": "FlagInSOSMain"}
            ]
            interrupts: list = info.get("interrupts", [])

        for action in actions:
            if not self.exec_main(context, action, interrupts):
                return CustomAction.RunResult(success=False)
        return CustomAction.RunResult(success=True)

    def exec_main(self, context: Context, action: dict | list, interrupts: list):
        retry_times = 0
        while retry_times < 20:
            # 先尝试执行主动作
            if self.exec_action(context.clone(), action):
                return True

            for interrupt in interrupts:
                context.tasker.controller.post_screencap().wait().get()
                if self.exec_action(context.clone(), interrupt):
                    retry_times = 0
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

                    # 先识别一下是否有选项界面
                    img = context.tasker.controller.post_screencap().wait().get()
                    check_reco = context.run_recognition("SOSSelectOption", img)
                    if not check_reco or not check_reco.best_result:
                        return False

                    pp_override = {"SOSSelectOption": {"interrupt": []}}

                    # 为每个 expected 创建独立节点
                    for i, expected in enumerate(expected_list):
                        node_name = f"SOSSelectOption_OCR_{i}"
                        # 基于 origin_node 创建新节点（使用深拷贝）
                        new_node = copy.deepcopy(origin_node)
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

                    # 先识别一下是否有选项界面
                    img = context.tasker.controller.post_screencap().wait().get()
                    check_reco = context.run_recognition("SOSSelectOption", img)
                    if not check_reco or not check_reco.best_result:
                        return False

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
            elif action_type == "SelectEncounterOption":
                method = action.get("method")
                if method == "OCR":
                    expected: str = action.get("expected", "")
                    order_by: str = action.get("order_by", "Vertical")

                    # 先识别一下是否有途中偶遇选项界面
                    img = context.tasker.controller.post_screencap().wait().get()
                    check_reco = context.run_recognition(
                        "SOSSelectEncounterOptionRec_Template", img
                    )
                    if not check_reco or not check_reco.best_result:
                        logger.debug("未识别到途中偶遇选项界面，跳过")
                        return False

                    context.run_task(
                        "SelectEncounterOption_OCR",
                        pipeline_override={
                            "SelectEncounterOption_OCR": {
                                "custom_action_param": {"expected": expected}
                            },
                            "SOSSelectEncounterOptionRec_Template": {
                                "order_by": order_by
                            },
                        },
                    )
                elif method == "HSV":
                    order_by: str = action.get("order_by", "Vertical")
                    index: int = action.get("index", 0)

                    # 先识别一下是否有途中偶遇选项界面
                    img = context.tasker.controller.post_screencap().wait().get()
                    check_reco = context.run_recognition(
                        "SOSSelectEncounterOptionRec_Template", img
                    )
                    if not check_reco or not check_reco.best_result:
                        logger.debug("未识别到途中偶遇选项界面，跳过")
                        return False

                    context.run_task(
                        "SOSSelectEncounterOption_HSV",
                        pipeline_override={
                            "SOSSelectEncounterOptionRec_Template": {
                                "order_by": order_by
                            }
                        },
                    )
                else:
                    logger.error(f"未知的途中偶遇选项选择方法: {method}")
                    return False
                return True
        return False


@AgentServer.custom_action("SOSSelectEncounterOption_OCR")
class SOSSelectEncounterOption_OCR(CustomAction):
    """
    局外演绎：无声综合征-途中偶遇选项内容识别-OCR版
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        expected: str = json.loads(argv.custom_action_param).get("expected")
        options: list[dict] = argv.reco_detail.raw_detail["best"]["detail"]["options"]

        for option in options:
            if expected in option["content"]:
                context.run_task(
                    "Click",
                    {
                        "Click": {
                            "action": "Click",
                            "target": option["roi"],
                            "post_delay": 1500,
                        }
                    },
                )
            return CustomAction.RunResult(success=True)
        return CustomAction.RunResult(success=False)


@AgentServer.custom_action("SOSSelectEncounterOption_HSV")
class SOSSelectEncounterOption_HSV(CustomAction):
    """
    局外演绎：无声综合征-途中偶遇选项内容识别-HSV版
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        index: int = json.loads(argv.custom_action_param).get(
            "SOSSelectEncounterOption_HSV", 0
        )
        options: list[dict] = argv.reco_detail.raw_detail["best"]["detail"]["options"]

        context.run_task(
            "Click",
            {
                "Click": {
                    "action": "Click",
                    "target": options[index]["roi"],
                    "post_delay": 1500,
                }
            },
        )
        return CustomAction.RunResult(success=True)


@AgentServer.custom_action("SOSShoppingList")
class SOSShoppingList(CustomAction):
    """
    局外演绎：无声综合征-购物列表处理
    """

    shopping_items: dict[str, int] = {}  # 存储识别到的物品 {name: price}

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        SOSShoppingList.shopping_items = {}

        # 加载物品数据用于纠错
        with open("resource/data/sos/items.json", encoding="utf-8") as f:
            items_data = json.load(f)

        # 构建所有有效物品名的集合（造物+谐波）
        valid_names = set()
        for type_items in items_data["artefacts"].values():
            valid_names.update(type_items)
        valid_names.update(items_data["harmonics"])

        all_items = {}  # 存储所有识别到的物品 {name: price}
        skipped_items = set()  # 存储所有跳过的物品名（已售出）
        last_results = []
        retry_times = 0

        while retry_times < 5:
            # 截图
            img = context.tasker.controller.post_screencap().wait().get()
            # 只保留接近黑色的像素，其他颜色都变成白色
            # 允许 RGB 每个通道在 0-95 范围内都认为是黑色
            mask = np.all(img <= 95, axis=-1)
            processed_img = np.where(mask[..., None], img, 255).astype(np.uint8)

            reco_detail = context.run_recognition("SOSShoppingListOCR", processed_img)
            if not reco_detail:
                retry_times += 1
                continue

            # 获取识别结果列表（已按垂直顺序排列）
            raw_detail = reco_detail.raw_detail
            current_results = raw_detail.get("filtered", []) if raw_detail else []

            # 配对物品名和价格（传入原始图像和context用于检测已售出标记）
            items, skipped = self._pair_items_and_prices(current_results, img, context)

            # 记录所有跳过的物品
            skipped_items.update(skipped)

            # 纠错并合并到总结果中
            for name, price in items.items():
                corrected_name = self._correct_item_name(name, valid_names)
                if corrected_name:
                    # 检查纠正后的名称是否在跳过列表中
                    if corrected_name in skipped_items:
                        logger.debug(
                            f"跳过已售出物品（纠错后匹配）: {name} -> {corrected_name}"
                        )
                        continue

                    # 如果纠正后的物品名已经在结果中，说明之前已经识别过
                    # 只保留价格更合理的那个（更小的价格，避免拼接价格）
                    if corrected_name in all_items:
                        # 保留价格更小的
                        if price < all_items[corrected_name]:
                            all_items[corrected_name] = price
                            logger.debug(
                                f"更新物品价格: {corrected_name} {all_items[corrected_name]} -> {price}"
                            )
                    else:
                        all_items[corrected_name] = price

            # 向下滑动
            context.run_task(
                "Swipe",
                {
                    "Swipe": {
                        "action": "Swipe",
                        "begin": [380, 459, 24, 21],
                        "end": [368, 120, 30, 27],
                        "duration": 500,
                        "post_delay": 800,
                    }
                },
            )

            # 判断是否划到底（本次识别结果和上次相同）
            if self._is_same_results(current_results, last_results):
                break

            last_results = current_results

            retry_times += 1

        logger.info(f"共识别到 {len(all_items)} 个可购买物品")
        for name, price in all_items.items():
            logger.info(f"{name}: {price}")

        # 存储到类静态变量
        SOSShoppingList.shopping_items = all_items

        return CustomAction.RunResult(success=True)

    def _pair_items_and_prices(
        self, results: list, img: np.ndarray, context: Context
    ) -> tuple[dict[str, int], set[str]]:
        """
        配对物品名和价格
        结果已按垂直顺序排列，价格在物品名下方约35-45像素处
        过滤掉已售出的物品（检测左上角"已售出"标记）

        返回: (物品字典, 跳过的物品名集合)
        """
        items = {}
        skipped = set()
        i = 0
        while i < len(results):
            current = results[i]
            current_text = current.get("text", "")
            current_box = current.get("box", [0, 0, 0, 0])
            current_y = current_box[1]

            # 判断是否为纯数字（价格）
            if current_text.isdigit():
                i += 1
                continue

            # 检查左上角是否有"已售出"标记
            # 使用 pipeline 识别"已售出"
            sold_out_reco = context.run_recognition(
                "SOSShoppingItemSoldOut",
                img,
                {
                    "SOSShoppingItemSoldOut": {
                        "roi": [
                            current_box[0] - 145,  # x: 物品名左上角往左扩展
                            current_box[1] - 17,  # y: 物品名左上角往上扩展
                            62,  # width
                            24,  # height
                        ]
                    }
                },
            )

            if sold_out_reco:
                logger.debug(f"跳过已售出物品: {current_text}")
                skipped.add(current_text)
                i += 1
                continue

            # 这是物品名，查找其对应的价格
            price = None
            if i + 1 < len(results):
                next_item = results[i + 1]
                next_text = next_item.get("text", "")
                next_y = next_item.get("box", [0, 0, 0, 0])[1]

                # 检查下一个是否为价格（纯数字且y坐标差在合理范围内）
                y_diff = next_y - current_y
                if next_text.isdigit() and 30 <= y_diff <= 50:
                    price_value = int(next_text)
                    # 过滤异常价格（防止拼接价格）
                    # 游戏中单个物品价格通常不超过1000
                    if price_value <= 1000:
                        price = price_value

            if price:
                items[current_text] = price

            i += 1

        return items, skipped

    def _correct_item_name(self, name: str, valid_names: set) -> str:
        """
        纠正识别错误的物品名
        使用编辑距离找到最相似的有效名称
        """
        if name in valid_names:
            return name

        # 计算与所有有效名称的相似度
        def edit_distance(s1: str, s2: str) -> int:
            """计算编辑距离"""
            if len(s1) < len(s2):
                return edit_distance(s2, s1)
            if len(s2) == 0:
                return len(s1)

            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row

            return previous_row[-1]

        # 找最相似的名称
        min_distance = float("inf")
        best_match = None

        for valid_name in valid_names:
            distance = edit_distance(name, valid_name)
            # 只接受距离小于名称长度一半的匹配
            if distance < min_distance and distance <= len(name) // 2:
                min_distance = distance
                best_match = valid_name

        if best_match:
            if best_match != name:
                logger.debug(f"纠正物品名: {name} -> {best_match}")
            return best_match

        logger.warning(f"未找到匹配的物品名: {name}")
        return name  # 返回原名称

    def _is_same_results(self, current: list, last: list) -> bool:
        """
        判断两次识别结果是否相同（通过文本内容比较）
        """
        if not last:
            return False

        current_texts = {item.get("text", "") for item in current}
        last_texts = {item.get("text", "") for item in last}

        # 如果有80%以上的内容相同，认为到底了
        if not current_texts or not last_texts:
            return False

        intersection = current_texts & last_texts
        return len(intersection) / len(current_texts) >= 0.8


@AgentServer.custom_action("SOSBuyItems")
class SOSBuyItems(CustomAction):
    """
    局外演绎：无声综合征-购买物品
    根据当前金雀子儿和商品列表，尽可能多地购买物品
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        # 识别右上角当前金雀子儿
        img = context.tasker.controller.post_screencap().wait().get()
        money_roi = [1125, 18, 88, 28]  # 右上角金雀子儿的ROI，需要根据实际调整

        reco_detail = context.run_recognition(
            "OCR",
            img,
            {"OCR": {"recognition": "OCR", "roi": money_roi, "expected": r"\d{1,5}"}},
        )

        if not reco_detail:
            logger.error("无法识别当前金雀子儿")
            return CustomAction.RunResult(success=False)

        ocr_result = cast(OCRResult, reco_detail.best_result)
        money_text = ocr_result.text

        # 提取数字
        money_match = re.search(r"\d+", money_text)
        if not money_match:
            logger.error(f"无法从文本中提取金雀子儿数量: {money_text}")
            return CustomAction.RunResult(success=False)

        current_money = int(money_match.group())
        logger.info(f"当前金雀子儿: {current_money}")

        # 获取购物清单
        shopping_items = SOSShoppingList.shopping_items
        if not shopping_items:
            logger.warning("购物清单为空")
            return CustomAction.RunResult(success=True)

        # 加载物品优先级配置（可选）
        try:
            with open("resource/data/sos/items.json", encoding="utf-8") as f:
                items_data = json.load(f)
            # 可以在这里定义优先级逻辑，暂时按价格升序排列（买便宜的，数量更多）
        except:
            pass

        # 贪心算法：按价格从低到高排序，尽可能购买更多数量
        sorted_items = sorted(
            shopping_items.items(), key=lambda x: x[1], reverse=False  # 价格从低到高
        )

        purchased_items = []
        remaining_money = current_money
        purchased_set = set()  # 已购买的物品名集合
        last_screen_texts = set()  # 上一屏的物品文本集合，用于判断是否到顶部

        # 从底部开始向上滑动购买
        max_scroll_times = 10
        scroll_times = 0

        while scroll_times < max_scroll_times:
            # 截图并识别当前屏幕的物品
            img = context.tasker.controller.post_screencap().wait().get()
            reco_detail = context.run_recognition("SOSShoppingListOCR", img)

            if not reco_detail:
                scroll_times += 1
                # 向上滑动
                context.run_task(
                    "Swipe",
                    {
                        "Swipe": {
                            "action": "Swipe",
                            "begin": [368, 120, 30, 27],
                            "end": [380, 459, 24, 21],
                            "duration": 500,
                            "post_delay": 500,
                        }
                    },
                )
                continue

            raw_detail = reco_detail.raw_detail
            current_results = raw_detail.get("filtered", []) if raw_detail else []

            # 获取当前屏幕的物品文本集合
            current_screen_texts = {
                item.get("text", "")
                for item in current_results
                if not item.get("text", "").isdigit()
            }

            # 判断是否到达顶部（与上一屏内容80%相同）
            if last_screen_texts:
                intersection = current_screen_texts & last_screen_texts
                if (
                    current_screen_texts
                    and len(intersection) / len(current_screen_texts) >= 0.8
                ):
                    break

            last_screen_texts = current_screen_texts

            # 使用黑色过滤后的图像再次识别，以检查价格是否可见（红色价格会被过滤）
            mask = np.all(img <= 95, axis=-1)
            processed_img = np.where(mask[..., None], img, 255).astype(np.uint8)
            price_reco_detail = context.run_recognition(
                "SOSShoppingListOCR", processed_img
            )

            # 获取可见价格的物品名集合（金雀子儿足够的物品）
            affordable_items = set()
            if price_reco_detail:
                price_raw_detail = price_reco_detail.raw_detail
                price_results = (
                    price_raw_detail.get("filtered", []) if price_raw_detail else []
                )

                # 配对物品名和价格，只有能配对成功的说明价格可见
                i = 0
                while i < len(price_results):
                    current = price_results[i]
                    current_text = current.get("text", "")

                    # 跳过纯数字
                    if current_text.isdigit():
                        i += 1
                        continue

                    # 检查下一个是否为价格
                    if i + 1 < len(price_results):
                        next_item = price_results[i + 1]
                        next_text = next_item.get("text", "")

                        if next_text.isdigit():
                            # 这个物品有价格，说明买得起
                            affordable_items.add(current_text)

                    i += 1

            # 在当前屏幕找到所有待购买的物品（必须在affordable_items中）
            items_on_screen = []
            for result in current_results:
                text = result.get("text", "")
                # 检查是否是待购买列表中的物品，且价格可见（买得起）
                for item_name, item_price in sorted_items:
                    if (
                        (item_name in text or text in item_name)
                        and item_name not in purchased_set
                        and item_price <= remaining_money
                        and text in affordable_items  # 必须价格可见
                    ):
                        items_on_screen.append((item_name, item_price, result))
                        break

            # 如果当前屏幕没有可购买的物品，滑动到下一页
            if not items_on_screen:
                scroll_times += 1
                # 向上滑动
                context.run_task(
                    "Swipe",
                    {
                        "Swipe": {
                            "action": "Swipe",
                            "begin": [368, 120, 30, 27],
                            "end": [380, 459, 24, 21],
                            "duration": 500,
                            "post_delay": 500,
                        }
                    },
                )
                continue

            # 购买当前屏幕的物品
            purchased_current_screen = False
            for item_name, item_price, result in items_on_screen:
                # 在购买前再次检查是否还买得起（因为购买其他物品后金钱可能减少）
                if item_price > remaining_money:
                    continue

                if self._buy_item_on_screen(context, item_name, result):
                    purchased_items.append((item_name, item_price))
                    purchased_set.add(item_name)
                    remaining_money -= item_price
                    purchased_current_screen = True
                    logger.info(
                        f"购买成功: {item_name} ({item_price}), 剩余: {remaining_money}"
                    )
                else:
                    logger.warning(f"购买失败: {item_name}")

            # 检查是否还有未购买的物品
            remaining_items = [
                (name, price)
                for name, price in sorted_items
                if name not in purchased_set and price <= remaining_money
            ]

            if not remaining_items:
                break

            # 如果当前屏幕购买了物品，重新检查当前屏幕，否则滑动
            if not purchased_current_screen:
                scroll_times += 1
                # 向上滑动
                context.run_task(
                    "Swipe",
                    {
                        "Swipe": {
                            "action": "Swipe",
                            "begin": [368, 120, 30, 27],
                            "end": [380, 459, 24, 21],
                            "duration": 500,
                            "post_delay": 500,
                        }
                    },
                )

        logger.info(f"购买完成，共购买 {len(purchased_items)} 件物品")
        logger.info(f"花费: {current_money - remaining_money}, 剩余: {remaining_money}")

        return CustomAction.RunResult(success=True)

    def _buy_item_on_screen(
        self, context: Context, item_name: str, result: dict
    ) -> bool:
        """
        购买当前屏幕上的指定物品
        """
        box = result.get("box", [0, 0, 0, 0])

        # 点击物品名称区域
        context.run_task(
            "Click",
            {
                "Click": {
                    "action": "Click",
                    "target": box,
                    "post_delay": 500,
                }
            },
        )

        # 确认左侧已选中该物品
        time.sleep(0.3)
        img = context.tasker.controller.post_screencap().wait().get()

        selected_reco = context.run_recognition(
            "SOSShoppingItemSelected",
            img,
            {"SOSShoppingItemSelected": {"roi": [box[0] - 150, box[1] - 6, 35, 100]}},
        )

        if not selected_reco:
            logger.warning(f"左侧未确认选中物品: {item_name}")
            return False

        # 点击右下角的购买按钮
        time.sleep(0.2)
        img = context.tasker.controller.post_screencap().wait().get()

        # 先检查是否已购买
        bought_roi = [1114, 647, 76, 35]
        bought_reco = context.run_recognition(
            "OCR",
            img,
            {"OCR": {"recognition": "OCR", "roi": bought_roi}},
        )

        if bought_reco:
            ocr_result = cast(OCRResult, bought_reco.best_result)
            button_text = ocr_result.text

            if "已购买" in button_text or "已购" in button_text:
                logger.info(f"物品已购买: {item_name}")
                return True

        # 检查购买按钮并点击
        buy_button_reco = context.run_recognition("SOSBuyButton", img)
        if buy_button_reco:
            # 最多重试5次购买
            buy_retry = 0
            while buy_retry < 5:
                context.run_task("SOSBuyButton")

                # 等待并确认购买成功（按钮变为"已购买"）
                time.sleep(0.5)
                img = context.tasker.controller.post_screencap().wait().get()
                confirm_reco = context.run_recognition(
                    "OCR",
                    img,
                    {"OCR": {"recognition": "OCR", "roi": bought_roi}},
                )

                if confirm_reco:
                    ocr_result = cast(OCRResult, confirm_reco.best_result)
                    confirm_text = ocr_result.text

                    if "已购买" in confirm_text or "已购" in confirm_text:
                        return True
                    else:
                        buy_retry += 1
                else:
                    buy_retry += 1

            logger.error(f"购买失败，已重试5次: {item_name}")
            return False
        else:
            logger.warning(f"未找到购买按钮")
            return False
