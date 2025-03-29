import os
import json
from datetime import datetime

from PIL import Image
from maa.agent.agent_server import AgentServer
from maa.custom_action import CustomAction
from maa.context import Context

from utils import logger


@AgentServer.custom_action("Screenshot")
class Screenshot(CustomAction):
    """
    自定义截图动作，保存当前屏幕截图到指定目录。

    参数格式:
    {
        "save_dir": "保存截图的目录路径"
    }
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        # image array(BGR)
        screen_array = context.tasker.controller.post_screencap().wait().get()

        # BGR2RGB
        if len(screen_array.shape) == 3 and screen_array.shape[2] == 3:
            rgb_array = screen_array[:, :, ::-1]
        else:
            rgb_array = screen_array
            logger.warning("当前截图并非三通道")

        img = Image.fromarray(rgb_array)

        save_dir = json.loads(argv.custom_action_param)["save_dir"]
        os.makedirs(save_dir, exist_ok=True)
        img.save(f"{save_dir}/{self._get_format_timestamp()}.png")
        logger.info(f"截图保存至 {save_dir}/{self._get_format_timestamp()}.png")

        return CustomAction.RunResult(success=True)

    def _get_format_timestamp(self):

        now = datetime.now()

        date = now.strftime("%Y.%m.%d")
        time = now.strftime("%H.%M.%S")
        milliseconds = f"{now.microsecond // 1000:03d}"

        return f"{date}-{time}.{milliseconds}"


@AgentServer.custom_action("DisableNode")
class DisableNode(CustomAction):
    """
    将特定 node 设置为 disable 状态 。

    参数格式:
    {
        "node_name": "结点名称"
    }
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        node_name = json.loads(argv.custom_action_param)["node_name"]

        context.override_pipeline({f"{node_name}": {"enabled": False}})

        return CustomAction.RunResult(success=True)


@AgentServer.custom_action("NodeOverride")
class NodeOverride(CustomAction):
    """
    在 node 中执行 pipeline_override 。

    参数格式:
    {
        "node_name": {"被覆盖参数": "覆盖值"},
        "node_name1": {"被覆盖参数": "覆盖值"}
    }
    """

    def run(
        self,
        context: Context,
        argv: CustomAction.RunArg,
    ) -> CustomAction.RunResult:

        nodes = json.loads(argv.custom_action_param)

        for node in nodes:
            for key in nodes[node]:
                context.override_pipeline({f"{node}": {f"{key}": nodes[node][key]}})

        return CustomAction.RunResult(success=True)
