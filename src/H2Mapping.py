from loggers import BasicLogger
from mapping import Mapping
from utils.import_util import get_dataset


class H2Mapping:
    def __init__(self, args):
        self.args = args
        # logger (optional)
        self.logger = BasicLogger(args)
        # data stream
        self.data_stream = get_dataset(args)
        # mapper
        self.mapper = Mapping(args, self.logger, data_stream=self.data_stream)
        # initialize map with first frame
        self.firstframe = self.mapper.initfirst_onlymap()

    def run(self):
        self.mapper.run(self.firstframe)
