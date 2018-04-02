import tkinter as tk

class Mazerunner(tk.Tk):
    '''
    Base GUI Class
    '''

    def __init__(self, *args, **kwargs):
        self._initalize_gui()

    def _initalize_gui(self):
        '''
        Set up the GUI and all needed parameters. Declares a shared container wich all additional frames have to
        inherit.
        '''

        super().__init__()
        self.title('MazeRunner')
        container = tk.Frame(self)
        container.pack(side='top', fill='both', expand=True)

        #standard tkinter configuration
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        frame = MainPage(container)
        frame.grid(row=0, column=0, sticky='nsew')

    def run(self):
        self.mainloop()

class MainPage(tk.Frame):
    '''
    This is the window wich will be shown to the user.
    It contains a start button, an image frame and the corresponding statistics.
    '''

    def __init__(self, parent):
        super().__init__(parent)


if __name__ == '__main__':
    maze_runner = Mazerunner()
    maze_runner.run()