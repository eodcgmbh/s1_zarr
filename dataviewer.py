import xarray as xr
import panel as pn
import matplotlib.pyplot as plt

pn.extension()

ds = xr.open_zarr("INCA.zarr")
chunk_data = ds["T2M"].isel(x=slice(0,100), y=slice(0,100)).load()

time_values = chunk_data.time.values
time_index = 0  # start time slice

mpl_pane = pn.pane.Matplotlib(sizing_mode='stretch_both')

def plot_time(ti):
    fig, ax = plt.subplots(figsize=(6,6))
    chunk_data.isel(time=ti).plot(ax=ax, vmin=-10, vmax=25)
    ax.set_title(f"Time: {time_values[ti]}")
    plt.close(fig)
    return fig

def update_plot(ti):
    mpl_pane.object = plot_time(ti)

update_plot(time_index)

# Create a Panel pane to catch mouse wheel events
class TimeScroll(pn.reactive.Reactive):
    time_idx = pn.param.Integer(time_index, bounds=(0, len(time_values)-1))

    def on_wheel(self, event):
        # event.delta is +1 or -1 (mouse wheel up/down)
        new_idx = self.time_idx - event.delta
        if 0 <= new_idx < len(time_values):
            self.time_idx = new_idx
            update_plot(self.time_idx)

time_scroll = TimeScroll()

# Bind param change to update plot
time_scroll.param.watch(lambda e: update_plot(e.new), 'time_idx')

# Panel layout
layout = pn.Column(
    pn.pane.Markdown("**Scroll mouse wheel on this text to change time slice**"),
    pn.panel(time_scroll, height=100, background='lightgrey'),
    mpl_pane,
    sizing_mode='stretch_width'
)

layout.servable()
layout.show()
