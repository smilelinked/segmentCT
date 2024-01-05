import asyncio
import logging

import tornado.web
from tornado.ioloop import IOLoop
from tornado.log import enable_pretty_logging

from ctsegment.algm import CTSegment
from handler import handler


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")


def make_app():
    settings = {
        "secret_key": "42wqTE23123wffLU94342wgldgFs",
    }

    return tornado.web.Application(
        [
            (r"/", MainHandler),
            (r"/ct_enqueue", handler.CTInQueueHandler),
        ],
        **settings
    )


async def main():
    app = make_app()
    app.listen(8086)
    shutdown_event = asyncio.Event()
    IOLoop.current().spawn_callback(CTSegment.segment)
    await shutdown_event.wait()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
                        filename='segment.log',
                        filemode='a',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                        # a是追加模式，默认如果不写的话，就是追加模式
                        format=
                        '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                        )
    enable_pretty_logging()
    asyncio.run(main())
