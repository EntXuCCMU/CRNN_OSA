import numpy as np
import scipy.signal


def apply_post_processing(frame_preds, frame_probs, duration_sec=60.0):
    """
    Implements Section 2.4: Post-processing Strategy
    1. Sequence Smoothing (Median Filter)
    2. Event Merging (Gap < 3s)
    3. Duration Filtering (Duration >= 10s)

    Args:
        frame_preds: List or array of frame-level class indices (0=Normal, 1=Hypopnea, 2=Apnea)
        duration_sec: Duration of the recording in seconds
    Returns:
        events: List of dicts [{'start': s, 'end': e, 'type': t}, ...]
    """

    # --- Step 1: Sequence Smoothing ---
    # "To mitigate salt-and-pepper noise, we applied a median filter (kernel size = 5)"
    smoothed_preds = scipy.signal.medfilt(frame_preds, kernel_size=5).astype(int)

    # Convert frame indices to time segments
    num_frames = len(smoothed_preds)
    time_per_frame = duration_sec / num_frames

    raw_events = []
    if num_frames == 0: return []

    # Simple Run-Length Encoding to get segments
    current_start = 0
    current_label = smoothed_preds[0]

    for i in range(1, num_frames):
        if smoothed_preds[i] != current_label:
            if current_label != 0:  # Ignore Normal
                raw_events.append({
                    'start': current_start * time_per_frame,
                    'end': i * time_per_frame,
                    'type': current_label
                })
            current_start = i
            current_label = smoothed_preds[i]
    # Add last event
    if current_label != 0:
        raw_events.append({
            'start': current_start * time_per_frame,
            'end': num_frames * time_per_frame,
            'type': current_label
        })

    if not raw_events:
        return []

    # --- Step 2: Event Merging ---
    # "Adjacent events of the same class separated by a gap of less than 3 seconds were merged"
    merged_events = []
    current_event = raw_events[0]

    for next_event in raw_events[1:]:
        gap = next_event['start'] - current_event['end']

        if (next_event['type'] == current_event['type']) and (gap < 3.0):
            # Merge
            current_event['end'] = next_event['end']
        else:
            merged_events.append(current_event)
            current_event = next_event
    merged_events.append(current_event)

    # --- Step 3: Duration Filtering ---
    # "Any candidate event with a duration shorter than 10 seconds ... was discarded"
    final_events = []
    for event in merged_events:
        duration = event['end'] - event['start']
        if duration >= 10.0:
            final_events.append(event)

    return final_events


def calculate_ahi(events, total_hours):
    """
    Implements Section 2.7.3: AHI Calculation
    AHI = (Total Apnea + Hypopnea) / Total Sleep Time (hours)
    """
    if total_hours <= 0: return 0.0
    count = len(events)
    return count / total_hours