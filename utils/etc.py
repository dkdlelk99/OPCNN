

def time_check(total_time):
    H = 60*60
    M = 60
    if total_time >= H:
        return f"{int(total_time//H)}h {(int(total_time-H)//M)}m {total_time%M:.2f}s"
    elif total_time >= M:
        return f"{(int(total_time)//M)}m {total_time%M:.2f}s"
    else:
        return f"{total_time%M:.2f}s"