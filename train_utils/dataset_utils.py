import traceback


def retry_get_example(func):
    def wrapper(self, i):
        try:
            return func(self, i)
        except Exception as e:
            traceback.print_exc(limit=0)
            return func(self, 1)
    return wrapper
