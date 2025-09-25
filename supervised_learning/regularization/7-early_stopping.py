#!/usr/bin/env python3


def early_stopping(cost, opt_cost, threshold, patience, count):

    if cost < opt_cost - threshold:
        count = 0
    else:
        count += 1

    if count >= patience:
        return True, count
    else:
        return False, count
