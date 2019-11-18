import threading

from cmd import Cmd


class InteractiveTrain(Cmd):

    prompt = ''

    def __init__(self, *args, **kwargs):
        self.bbox_plotter = kwargs.pop('bbox_plotter', None)
        self.optimizer = kwargs.pop('optimizer', None)
        self.optimizer_member_name = kwargs.pop('optmizer_member', 'alpha')
        super().__init__(*args, **kwargs)

    def do_enablebboxvis(self, arg):
        """Enable sending of bboxes to remote host"""
        if self.bbox_plotter is not None:
            self.bbox_plotter.send_bboxes = True

    def do_shiftlr(self, arg):
        shift =float(arg)
        if self.optimizer is not None:
            if hasattr(self.optimizer, '__iter__'):
                for optimizer in self.optimizer:
                    hyperparam = optimizer.hyperparam
                    setattr(hyperparam, self.optimizer_member_name, getattr(hyperparam, self.optimizer_member_name) * shift)
                    print(f"setting learning rate to: {getattr(hyperparam, self.optimizer_member_name)}")
            else:
                hyperparam = self.optimizer.hyperparam
                setattr(hyperparam, self.optimizer_member_name, getattr(hyperparam, self.optimizer_member_name) * shift)
                print(f"setting learning rate to: {getattr(hyperparam, self.optimizer_member_name)}")

    def do_quit(self, arg):
        return True

    def do_echo(self, arg):
        print(arg)


def open_interactive_prompt(*args, **kwargs):
    cmd_interface = InteractiveTrain(*args, **kwargs)

    thread = threading.Thread(target=lambda: cmd_interface.cmdloop())
    thread.daemon = True
    thread.start()
