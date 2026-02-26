def build_report(bsor_obj, label):
    notes = getattr(bsor_obj, "notes", []) or []

    left = []
    right = []

    for n in notes:
        try:
            saber = getattr(n, "colorType", None)
            pre_score = getattr(n, "pre_score", None)
            post_score = getattr(n, "post_score", None)
            acc_score = getattr(n, "acc_score", None)
            time = getattr(n, "event_time", None)

            entry = {
                "time": time,
                "pre_score": pre_score,
                "post_score": post_score,
                "acc_score": acc_score,
            }

            if saber == 0:
                left.append(entry)
            elif saber == 1:
                right.append(entry)
        except:
            continue

    def avg(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    def pct_under(xs, threshold):
        xs = [x for x in xs if x is not None]
        if not xs:
            return None
        return sum(1 for x in xs if x < threshold) / len(xs)

    def swing_from_pre(pre):
        return (pre / 70) * 100 if pre is not None else None

    def swing_from_post(post):
        return (post / 30) * 60 if post is not None else None

    def summarize(hand):
        pre = [h["pre_score"] for h in hand]
        post = [h["post_score"] for h in hand]
        acc = [h["acc_score"] for h in hand]

        return {
            "count": len(hand),
            "pre_score_avg": avg(pre),
            "post_score_avg": avg(post),
            "acc_score_avg": avg(acc),

            "pre_swing_deg_avg": avg([swing_from_pre(p) for p in pre]),
            "post_swing_deg_avg": avg([swing_from_post(p) for p in post]),

            "underswing_pre_rate": pct_under(pre, 70),
            "underswing_post_rate": pct_under(post, 30),
        }

    report = {
        "app": {"name": APP_NAME, "version": APP_VERSION},
        "meta": {
            "label": label,
            "parsed_at": datetime.datetime.now().isoformat(),
            "notes_total": len(notes),
        },
        "summary": {
            "left": summarize(left),
            "right": summarize(right),
            "all": summarize(left + right),
        },
        "notes": "Swing angles are derived from Beat Saber scoring caps (70 pre / 30 post).",
    }

    return report
