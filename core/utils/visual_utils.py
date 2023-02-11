import uuid
import torch
import numpy as np
import tensorflow as tf
import math

from IPython.display import HTML
import matplotlib.animation as animation
import matplotlib.pyplot as plt



def create_figure_and_axes(size_pixels):
    """Initializes a unique figure and axes for plotting."""
    fig, ax = plt.subplots(1, 1, num=uuid.uuid4())

    # Sets output image to pixel resolution.
    dpi = 100
    size_inches = size_pixels / dpi
    fig.set_size_inches([size_inches, size_inches])
    fig.set_dpi(dpi)
    fig.set_facecolor('white')
    ax.set_facecolor('white')
    ax.xaxis.label.set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.yaxis.label.set_color('black')
    ax.tick_params(axis='y', colors='black')
    fig.set_tight_layout(True)
    ax.grid(False)
    return fig, ax


def fig_canvas_image(fig):
    """
    Returns a [H, W, 3] uint8 np.array image from fig.canvas.tostring_rgb().
    """
    # Just enough margin in the figure to display xticks and yticks.
    fig.subplots_adjust(
        left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def get_colormap(num_agents):
    """
    Compute a color map array of shape [num_agents, 4].
    """
    colors = plt.cm.get_cmap('jet', num_agents)
    colors = colors(range(num_agents))
    np.random.shuffle(colors)
    return colors


def get_viewport(all_states, all_states_mask):
    """
    Gets the region containing the data.

    Args:
        all_states: states of agents as an array of shape [num_agents, num_steps,
        2].
        all_states_mask: binary mask of shape [num_agents, num_steps] for
        `all_states`.

    Returns:
        center_y: float. y coordinate for center of data.
        center_x: float. x coordinate for center of data.
        width: float. Width of data.
    """
    valid_states = all_states[all_states_mask]
    all_y = valid_states[..., 1]
    all_x = valid_states[..., 0]

    center_y = (np.max(all_y) + np.min(all_y)) / 2
    center_x = (np.max(all_x) + np.min(all_x)) / 2

    range_y = np.ptp(all_y)
    range_x = np.ptp(all_x)

    width = max(range_y, range_x)

    return center_y, center_x, width


def visualize_one_step(
    states,
    mask,
    roadgraph,
    title,
    center_y,
    center_x,
    width,
    color_map,
    size_pixels=1000):
    """
    Generate visualization for a single step.
    """

    # Create figure and axes.
    fig, ax = create_figure_and_axes(size_pixels=size_pixels)

    # Plot roadgraph.
    rg_pts = roadgraph[:, :2].T
    ax.plot(rg_pts[0, :], rg_pts[1, :], 'k.', alpha=1, ms=2)

    masked_x = states[:, 0][mask]
    masked_y = states[:, 1][mask]
    colors = color_map[mask]

    # Plot agent current position.
    ax.scatter(
        masked_x,
        masked_y,
        marker='o',
        linewidths=3,
        color=colors,
    )

    # Title.
    ax.set_title(title)

    # Set axes.  Should be at least 10m on a side.
    size = max(10, width * 1.0)
    ax.axis([
        -size / 2 + center_x, size / 2 + center_x, -size / 2 + center_y,
        size / 2 + center_y
    ])
    ax.set_aspect('equal')

    image = fig_canvas_image(fig)
    plt.close(fig)
    return image


def visualize_all_agents_smooth(decoded_example, size_pixels=1000):
    """
    Visualizes all agent predicted trajectories in a serie of images.

    Args:
        decoded_example: Dictionary containing agent info about all modeled agents.
        size_pixels: The size in pixels of the output image.

    Returns:
        T of [H, W, 3] uint8 np.arrays of the drawn matplotlib's figure canvas.
    """
    # [num_agents, num_past_steps, 2] float32.
    past_states = tf.stack(
        [decoded_example['state/past/x'], decoded_example['state/past/y']],
        -1).numpy()
    past_states_mask = decoded_example['state/past/valid'].numpy() > 0.0

    # [num_agents, 1, 2] float32.
    current_states = tf.stack(
        [decoded_example['state/current/x'], decoded_example['state/current/y']],
        -1).numpy()
    current_states_mask = decoded_example['state/current/valid'].numpy() > 0.0

    # [num_agents, num_future_steps, 2] float32.
    future_states = tf.stack(
        [decoded_example['state/future/x'], decoded_example['state/future/y']],
        -1).numpy()
    future_states_mask = decoded_example['state/future/valid'].numpy() > 0.0

    # [num_points, 3] float32.
    roadgraph_xyz = decoded_example['roadgraph_samples/xyz'].numpy()

    num_agents, num_past_steps, _ = past_states.shape
    num_future_steps = future_states.shape[1]

    color_map = get_colormap(num_agents)

    # [num_agents, num_past_steps + 1 + num_future_steps, depth] float32.
    all_states = np.concatenate([past_states, current_states, future_states], 1)

    # [num_agents, num_past_steps + 1 + num_future_steps] float32.
    all_states_mask = np.concatenate(
        [past_states_mask, current_states_mask, future_states_mask], 1)

    center_y, center_x, width = get_viewport(all_states, all_states_mask)

    images = []

    # Generate images from past time steps.
    for i, (s, m) in enumerate(
        zip(
            np.split(past_states, num_past_steps, 1),
            np.split(past_states_mask, num_past_steps, 1))):
        im = visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz,
                                'past: %d' % (num_past_steps - i), center_y,
                                center_x, width, color_map, size_pixels)
        images.append(im)

    # Generate one image for the current time step.
    s = current_states
    m = current_states_mask

    im = visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz, 'current', center_y,
                            center_x, width, color_map, size_pixels)
    images.append(im)

    # Generate images from future time steps.
    for i, (s, m) in enumerate(
        zip(
            np.split(future_states, num_future_steps, 1),
            np.split(future_states_mask, num_future_steps, 1))):
        im = visualize_one_step(s[:, 0], m[:, 0], roadgraph_xyz,
                                'future: %d' % (i + 1), center_y, center_x, width,
                                color_map, size_pixels)
    images.append(im)

    return images

def create_animation(images, interval=100):
    """ 
    Creates a Matplotlib animation of the given images.

    Args:
        images: A list of numpy arrays representing the images.
        interval: Delay between frames in milliseconds.

    Returns:
        A matplotlib.animation.Animation.

    Usage:
        anim = create_animation(images)
        anim.save('/tmp/animation.avi')
        HTML(anim.to_html5_video())
    """

    plt.ioff()
    fig, ax = plt.subplots()
    dpi = 100
    size_inches = 1000 / dpi
    fig.set_size_inches([size_inches, size_inches])
    plt.ion()

    def animate_func(i):
        ax.imshow(images[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid('off')

    anim = animation.FuncAnimation(
        fig, animate_func, frames=len(images), interval=interval)
    plt.close(fig)
    return anim

    
def occupancy_rgb_image(agent_grids, roadgraph_image, gamma: float = 1.6):
    """
    Visualize predictions or ground-truth occupancy.
    Args:
        agent_grids: AgentGrids object containing optional
        vehicles/pedestrians/cyclists.
        roadgraph_image: Road graph image [batch_size, height, width, 1] float32.
        gamma: Amplify predicted probabilities so that they are easier to see.
    Returns:
        [batch_size, height, width, 3] float32 RGB image.
    """
    zeros = torch.zeros_like(roadgraph_image)
    ones  = torch.ones_like(zeros)

    agents = agent_grids
    veh = zeros if agents['vehicles'] is None else agents['vehicles'] # torch.permute(agents['vehicles'], (0, 3, 1, 2)) #torch.squeeze(agents['vehicles'], -1)
    ped = zeros if agents['pedestrians'] is None else agents['pedestrians']
    cyc = zeros if agents['cyclists'] is None else agents['cyclists']

    veh = torch.pow(veh, 1 / gamma)
    ped = torch.pow(ped, 1 / gamma)
    cyc = torch.pow(cyc, 1 / gamma)

    # Convert layers to RGB.
    rg_rgb  = torch.concat([zeros, zeros, zeros], axis=-1)
    veh_rgb = torch.concat([veh, zeros, zeros], axis=-1)  # Red.
    ped_rgb = torch.concat([zeros, ped * 0.67, zeros], axis=-1)  # Green.
    cyc_rgb = torch.concat([cyc * 0.33, zeros, zeros * 0.33], axis=-1)  # Purple.
    bg_rgb  = torch.concat([ones, ones, ones], axis=-1)  # White background.
    # Set alpha layers over all RGB channels.
    rg_a  = torch.concat([roadgraph_image, roadgraph_image, roadgraph_image], axis=-1)
    veh_a = torch.concat([veh, veh, veh], axis=-1)
    ped_a = torch.concat([ped, ped, ped], axis=-1)
    cyc_a = torch.concat([cyc, cyc, cyc], axis=-1)
    # Stack layers one by one.
    img, img_a = _alpha_blend(fg=rg_rgb, bg=bg_rgb, fg_a=rg_a)
    img, img_a = _alpha_blend(fg=veh_rgb, bg=img, fg_a=veh_a, bg_a=img_a)
    img, img_a = _alpha_blend(fg=ped_rgb, bg=img, fg_a=ped_a, bg_a=img_a)
    img, img_a = _alpha_blend(fg=cyc_rgb, bg=img, fg_a=cyc_a, bg_a=img_a)
    return img


def _alpha_blend(fg, bg, fg_a = None, bg_a = None):
    """
    Overlays foreground and background image with custom alpha values.
    Implements alpha compositing using Porter/Duff equations.
    https://en.wikipedia.org/wiki/Alpha_compositing
    Works with 1-channel or 3-channel images.
    If alpha values are not specified, they are set to the intensity of RGB
    values.
    Args:
        fg: Foreground: float32 tensor shaped [batch, grid_height, grid_width, d].
        bg: Background: float32 tensor shaped [batch, grid_height, grid_width, d].
        fg_a: Foreground alpha: float32 tensor broadcastable to fg.
        bg_a: Background alpha: float32 tensor broadcastable to bg.
    Returns:
        Output image: tf.float32 tensor shaped [batch, grid_height, grid_width, d].
        Output alpha: tf.float32 tensor shaped [batch, grid_height, grid_width, d].
    """
    if fg_a is None:
        fg_a = fg
    if bg_a is None:
        bg_a = bg
    eps = 1e-10
    out_a = fg_a + bg_a * (1 - fg_a)
    out_rgb = (fg * fg_a + bg * bg_a * (1 - fg_a)) / (out_a + eps)
    return out_rgb, out_a

def get_observed_occupancy_at_waypoint(waypoints, k: int):
    """
    Returns occupancies of currently-observed agents at waypoint k.
    """
    agent_grids = {'vehicles': None, 'pedestrians': None, 'cyclists' : None}
    if waypoints['vehicles']['observed_occupancy']:
      agent_grids['vehicles'] = waypoints['vehicles']['observed_occupancy'][k]
    if waypoints['pedestrians'] and waypoints['pedestrians']['observed_occupancy']:
      agent_grids['pedestrians'] = waypoints['pedestrians']['observed_occupancy'][k]
    if waypoints['cyclists'] and waypoints['cyclists']['observed_occupancy']:
      agent_grids['cyclists'] = waypoints['cyclists']['observed_occupancy'][k]
    return agent_grids

def get_occluded_occupancy_at_waypoint(waypoints, k: int):
    """
    Returns occupancies of currently-occluded agents at waypoint k.
    """
    agent_grids = {'vehicles': None, 'pedestrians': None, 'cyclists' : None}
    if waypoints['vehicles']['occluded_occupancy']:
      agent_grids['vehicles'] = waypoints['vehicles']['occluded_occupancy'][k]
    if waypoints['pedestrians'] and waypoints['pedestrians']['occluded_occupancy']:
      agent_grids['pedestrians'] = waypoints['pedestrians']['occluded_occupancy'][k]
    if waypoints['cyclists'] and waypoints['cyclists']['occluded_occupancy']:
      agent_grids['cyclists'] = waypoints['cyclists']['occluded_occupancy'][k]
    return agent_grids

def get_flow_at_waypoint(waypoints, k: int):
    """
    Returns flow fields of all agents at waypoint k.
    """
    agent_grids = {'vehicles': None, 'pedestrians': None, 'cyclists' : None}
    if waypoints['vehicles']['flow']:
      agent_grids['vehicles'] = waypoints['vehicles']['flow'][k]
    if waypoints['pedestrians'] and waypoints['pedestrians']['flow']:
      agent_grids['pedestrians'] = waypoints['pedestrians']['flow'][k]
    if waypoints['cyclists'] and waypoints['cyclists']['flow']:
      agent_grids['cyclists'] = waypoints['cyclists']['flow'][k]
    return agent_grids


def flow_rgb_image(flow, roadgraph_image, agent_trails=None):
    """
    Converts (dx, dy) flow to RGB image.
    Args:
        flow: [batch_size, height, width, 2] float32 tensor holding (dx, dy) values.
        roadgraph_image: Road graph image [batch_size, height, width, 1] float32.
        agent_trails: [batch_size, height, width, 1] float32 tensor containing
        rendered trails for all agents over the past and current time frames.
    Returns:
        [batch_size, height, width, 3] float32 RGB image.
    """
    # Swap x, y for compatibilty with published visualizations.
    flow = torch.roll(flow, shifts=1, dims=-1)
    # saturate_magnitude=-1 normalizes highest intensity to largest magnitude.
    flow_image = _optical_flow_to_rgb(flow, saturate_magnitude=-1)
    # Add roadgraph.
    flow_image = _add_grayscale_layer(roadgraph_image, flow_image)  # Black.
    # Overlay agent trails.
    # flow_image = _add_grayscale_layer(agent_trails * 0.2, flow_image)  # 0.2 alpha

    return flow_image


def _add_grayscale_layer(fg_a, scene_rgb):
    """
    Adds a black/gray layer using fg_a as alpha over an RGB image."""
    # Create a black layer matching dimensions of fg_a.
    black = torch.zeros_like(fg_a)
    black = torch.concat([black, black, black], axis=-1)
    # Add the black layer with transparency over the scene_rgb image.
    overlay, _ = _alpha_blend(fg=black, bg=scene_rgb, fg_a=fg_a, bg_a=1.0)
    return overlay

ONE = torch.ones((1)).to('cuda:0')

def _optical_flow_to_hsv(flow, saturate_magnitude = -1.0, name=None):
    """
    Visualize an optical flow field in HSV colorspace.
    This uses the standard color code with hue corresponding to direction of
    motion and saturation corresponding to magnitude.
    The attr `saturate_magnitude` sets the magnitude of motion (in pixels) at
    which the color code saturates. A negative value is replaced with the maximum
    magnitude in the optical flow field.

    Args:
        flow: A `Tensor` of type `float32`. A 3-D or 4-D tensor storing (a batch of)
        optical flow field(s) as flow([batch,] i, j) = (dx, dy). The shape of the
        tensor is [height, width, 2] or [batch, height, width, 2] for the 4-D
        case.
        saturate_magnitude: An optional `float`. Defaults to `-1`.
        name: A name for the operation (optional).

    Returns:
        An tf.float32 HSV image (or image batch) of size [height, width, 3]
        (or [batch, height, width, 3]) compatible with tensorflow color conversion
        ops. The hue at each pixel corresponds to direction of motion. The
        saturation at each pixel corresponds to the magnitude of motion relative to
        the `saturate_magnitude` value. Hue, saturation, and value are in [0, 1].
    """

    with tf.name_scope(name or 'OpticalFlowToHSV'):
        
        flow_shape = flow.shape
        if len(flow_shape) < 3:
            raise ValueError('flow must be at least 3-dimensional, got' f' `{flow_shape}`')
        if flow_shape[-1] != 2:
            raise ValueError(f'flow must have innermost dimension of 2, got' f' `{flow_shape}`')

        height = flow_shape[-3]
        width = flow_shape[-2]
        flow_flat = torch.reshape(flow, (-1, height, width, 2))

        dx = flow_flat[..., 0]
        dy = flow_flat[..., 1]
        # [batch_size, height, width]
        magnitudes = torch.sqrt(torch.square(dx) + torch.square(dy))
        if saturate_magnitude < 0:
            # [batch_size, 1, 1]
            local_saturate_magnitude = torch.amax(magnitudes, dim=(1, 2), keepdim=True)
        else:
            local_saturate_magnitude = torch.tensor(saturate_magnitude)

        # Hue is angle scaled to [0.0, 1.0).
        hue = (torch.remainder(torch.atan2(dy, dx), (2 * math.pi))) / (2 * math.pi)
        # Saturation is relative magnitude.
        relative_magnitudes = torch.div(magnitudes, local_saturate_magnitude)
        saturation = torch.minimum(relative_magnitudes, ONE)  # Larger magnitudes saturate.
        
        # Value is fixed.
        value = torch.ones_like(saturation)
        hsv_flat = torch.stack((hue, saturation, value), axis=-1)

        return torch.reshape(hsv_flat, (256, 256, 3))   # TODO dont hardcore grid size

def _optical_flow_to_rgb(
    flow, saturate_magnitude= -1.0,name= None):
    """
    Visualize an optical flow field in RGB colorspace.
    """
    name = name or 'OpticalFlowToRGB'
    hsv = _optical_flow_to_hsv(flow, saturate_magnitude, name)
    hsv = hsv[None, :]
    hsv = torch.permute(hsv, (0, 3, 1, 2))
    hsv = hsv2rgb(hsv)
    hsv = torch.permute(hsv, (0, 2, 3, 1))
    return hsv


def hsv2rgb(input):
    assert(input.shape[1] == 3)

    h, s, v = input[:, 0], input[:, 1], input[:, 2]
    h_ = (h - torch.floor(h / 360) * 360) / 60
    c = s * v
    x = c * (1 - torch.abs(torch.fmod(h_, 2) - 1))

    zero = torch.zeros_like(c)
    y = torch.stack((
        torch.stack((c, x, zero), dim=1),
        torch.stack((x, c, zero), dim=1),
        torch.stack((zero, c, x), dim=1),
        torch.stack((zero, x, c), dim=1),
        torch.stack((x, zero, c), dim=1),
        torch.stack((c, zero, x), dim=1),
    ), dim=0)

    index = torch.repeat_interleave(torch.floor(h_).unsqueeze(1), 3, dim=1).unsqueeze(0).to(torch.long)
    rgb = (y.gather(dim=0, index=index) + (v - c)).squeeze(0)
    return rgb