import tkinter as tk
import time
from PIL import Image, ImageTk

class Mazerunner(tk.Tk):
    '''
    Base GUI Class
    '''

    def __init__(self, train_func, *args, **kwargs):
        self._initalize_gui(train_func)

    def _initalize_gui(self, train_func):
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

        self.frame = MainPage(container, train_func)
        self.frame.grid(row=0, column=0, sticky='nsew')

    def run(self):
        self.mainloop()

class MainPage(tk.Frame):
    '''
    This is the window wich will be shown to the user.
    It contains a start button, an image frame and the corresponding statistics.
    '''

    def __init__(self, parent, train_func):
        '''

        :param parent:
        '''
        super().__init__(parent)
        #static elements
        tk.Label(self, text='Current status:').grid(row=0, column=0, sticky='nsew')
        tk.Label(self, text='Fitness of current model:').grid(row=1, column=0)
        tk.Button(self, text='Start training.', command=train_func).grid(row=6)
        tk.Label(self, text='Laps of current model:').grid(row=2, column=0)
        tk.Label(self, text='------------------------------').grid(row=3)
        tk.Label(self, text='Generation no.').grid(row=4, column=0)
        tk.Label(self, text='Best global fitness:').grid(row=5, column=0)

        #dynamic elements
        self.traveled_distance = tk.Label(self, text=0)
        self.traveled_distance.grid(row=1, column=1)
        first_image = Image.new('RGB', (100,100), color='black')
        first_image= ImageTk.PhotoImage(first_image)

        self.current_status=tk.Label(self, image=first_image)
        self.current_status.image = first_image
        self.current_status.grid(row=0, column=1)

        self.laps = tk.Label(self, text=0)
        self.laps.grid(row=2, column=1)

        self.generation_no = tk.Label(self, text=0)
        self.generation_no.grid(row=4, column=1)

        self.global_fitness = tk.Label(self, text=0)
        self.global_fitness.grid(row=5, column=1)

        #resize configurations for each row and column
        for row_or_col in range(0,7):
            self.grid_rowconfigure(row_or_col, weight=1)
            self.grid_columnconfigure(row_or_col, weight=1)

    def update_state(self, image, distance, laps, global_fitness, generation_no):
        '''
        Shows the given image in the image label of the GUI, updates distance and laps.
        :param image: The image to display
        '''
        image = image.resize((500,500))
        image = ImageTk.PhotoImage(image)
        self.current_status['image'] = image
        self.current_status.image=image

        self.traveled_distance['text'] = distance
        self.laps['text'] = laps
        self.global_fitness['text'] = global_fitness
        self.generation_no['text'] = generation_no
        self.update()