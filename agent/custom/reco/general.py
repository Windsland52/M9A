from typing import Union, Optional

from skimage.metrics import structural_similarity
from maa.agent.agent_server import AgentServer
from maa.custom_recognition import CustomRecognition
from maa.context import Context
from maa.define import RectType

from utils import logger


@AgentServer.custom_recognition("IsSimilar")
class IsSimilar(CustomRecognition):

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> Union[CustomRecognition.AnalyzeResult, Optional[RectType]]:

        context.tasker.controller.post_screencap().wait().get()

        print(argv)

        return CustomRecognition.AnalyzeResult(box=[0, 0, 0, 0], detail="test")

    def compare_ssim(img1, img2, multichannel=True):
        # 对于彩色图像需设置multichannel=True
        ssim = structural_similarity(img1, img2, multichannel=multichannel)
        return ssim
