import os
import json
import time
import numpy as np
# from threading import Thread

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.chrome.options import Options


import PIL

import ipdb


class MiniWoBInstance:
    """
    Interface to interact with a selenium broswer instance
    """

    WINDOW_WIDTH = 500
    WINDOW_HEIGHT = 240
    TASK_WIDTH = 160
    TASK_HEIGHT = 210

    def __init__(
            self, task_file,
            base_url=os.getenv("WOB_PATH"),
            wait_ms=0., block_on_reset=True, refresh_freq=0
            ):
        """
        E.g. base_url='file:///h/sheng/DOM-Q-NET/miniwob/html/miniwob/',
        Args:
            wait_ms: pause the instance after each action for this ms 
            block_on_reset: On reset, block until the page loads
            refresh_freq: Every this # episodes, refresh the page at the begin
                          of the next episode
        http://guppy7:8000/miniwob/
        """
        super(MiniWoBInstance, self).__init__()
        print("Opening MiniWoB")

        #url = base_url + task_file
        #options = webdriver.FirefoxOptions()
        #options.set_headless()
        #options.add_argument('-safe-mode')
        #self._driver = webdriver.Firefox(options=options)
        #self._driver.get(url)
        # print("Firefox Task title: " + self._driver.title)

        #NOTE Chrome driver
        url = base_url + task_file
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        self._url = url
        self._driver = webdriver.Chrome(chrome_options=chrome_options)
        self._driver.get(url)
        print("Chrome Task title: " + self._driver.title)

    def __del__(self):
        print("Closing MiniWoB")
        self._driver.quit()
        # self._driver.close()

    @property
    def utterance(self):
        return self._driver.execute_script('return core.getUtterance();')

    @property
    def dom(self):
        return self._driver.execute_script('return core.getDOMInfo();')

    @property
    def reward_avg(self):
        return self._driver.execute_script('return core.rewardAvg();')

    @property
    def is_done(self):
        return self._driver.execute_script(
                'return {'
                '"done": WOB_DONE_GLOBAL,'
                '};')

    @property
    def img(self):
        png_data = self._driver.get_screenshot_as_png()
        pil_img = PIL.Image.open(png_data)
        pil_img = pil_img.crop(
                (0, 0, self.TASK_WIDTH, self.TASK_HEIGHT)
                ).convert('RGB')
        return pil_img

    @property
    def metadata(self):
        return self._driver.execute_script(
                'return {'
                '"done": WOB_DONE_GLOBAL,'
                '"env_reward": WOB_REWARD_GLOBAL,'
                '"raw_reward": WOB_RAW_REWARD_GLOBAL,'
                '"info": WOB_REWARD_REASON,'
                '};')

    def begin_task(self, seed=None):
        """
        args:
            seed: e.g. 'hello', 'itsme',
        """
        seed=None
        if seed is not None:
            self._driver.execute_script('Math.seedrandom({});'.format(repr(seed)))
        # print(self._driver.execute_script('return WOB_TASK_READY;') )
        self._driver.execute_script('core.startEpisodeReal();')

    def force_stop(self):
        self._driver.execute_script('return core.endEpisode(0);')

    def terminate(self):
        self._driver.execute_script('return core.endEpisode(-1, false, "terminate");')

    def coord_click(self, left, top):
        body = self._driver.find_element_by_tag_name('body')
        chain = ActionChains(self._driver)
        chain.move_to_element_with_offset(body, left, top).click().perform()

    def dom_click(self, ref, fail_hard=False):
        result = self._driver.execute_script(
                'return core.elementClick({});'.format(ref)
                )
        if not result:
            if fail_hard:
                raise RuntimeError()
            else:
                pass

    def type(self, text): 
        # TODO WHY WOULD CLICK PAD TOKEN????
        chain = ActionChains(self._driver)
        chain.send_keys(text)
        chain.perform()
        if text == "<pad>":
            ipdb.set_trace()

    def focus_and_type(self, ref, text):
        self.dom_click(ref)
        self.type(text)


if __name__ == '__main__':
    pass

