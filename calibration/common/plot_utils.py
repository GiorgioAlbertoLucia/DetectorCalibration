from dataclasses import dataclass
from ROOT import TCanvas, TPaveText, TLegend

class PdfHandler:

    def __init__(self, pdf_output_path:str):
        
        self._pdf_path = pdf_output_path
        self._canvas = TCanvas('', '', 900, 600)
        self._canvas.Print(f'{self._pdf_path}(')

    @property
    def canvas(self):
        return self._canvas
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._canvas.Clear()
        self._canvas.Print(f'{self._pdf_path})')

    def cd(self):
        self._canvas.cd()

    def load_canvas(self, canvas):
        self._canvas = canvas
    
    def save(self):
        self._canvas.Print(self._pdf_path)

    def clear(self):
        self._canvas.Clear()

    def save_and_clear(self):
        self._canvas.Print(self._pdf_path)
        self._canvas.Clear()

    def draw_save_and_clear(self, obj, **kwargs):
        self.canvas_settings(**kwargs)
        self._canvas.cd()
        obj.Draw(kwargs.get('draw_option', ''))
        self._canvas.Print(self._pdf_path)
        self._canvas.Clear()

    def draw_and_save(self, obj, **kwargs):
        self.canvas_settings(**kwargs)
        self._canvas.cd()
        obj.Draw(kwargs.get('draw_option', ''))
        self._canvas.Print(self._pdf_path)
    
    def canvas_settings(self, **kwargs):
        self._canvas.SetLogy(kwargs.get('logy', False))
