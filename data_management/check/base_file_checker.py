from abc import abstractmethod
from email.headerregistry import Address


class BaseFileCheck(object):
    def __init__(self):
        self.file_folder = None
        self.subject_email = None
        self.email_recipients = None
        self.email_sender = Address("PAGER Bot", "pagerbot", "gfz-potsdam.de")

    @abstractmethod
    def check_files_exists(self, *args):
        raise NotImplementedError

    @abstractmethod
    def check_file_format(self, *args):
        raise NotImplementedError

    @abstractmethod
    def run_check(self, *args):
        raise NotImplementedError
