def _adjust_scale(pred: float, answer: float, scale: str):
    if round(pred, 2) == round(answer, 2):
        return True
    
    elif scale == 'percent' and round(pred * 100, 2) == round(answer, 2):
        return True

    else:
        return False


def _adjust_sum(nums: list, answer: float, scale: str):
    pred = sum(nums)
    return 1 if _adjust_scale(pred, answer, scale) else None


def _adjust_mul(nums: list, answer: float, scale: str):
    pred = 1

    for num in nums:
        pred *= num
    
    return 1 if _adjust_scale(pred, answer, scale) else None


def _adjust_avg(nums: list, answer: float, scale: str):
    if len(nums):
        pred = sum(nums) / len(nums)
        return 1 if _adjust_scale(pred, answer, scale) else None
    
    else:
        return None


def _adjust_diff(nums: list, answer: float, scale: str):
    if len(nums) != 2:
        return None

    elif _adjust_scale(nums[0] - nums[1], answer, scale) is not None:
        return 1
    
    elif _adjust_scale(nums[1] - nums[0], answer, scale) is not None:
        return -1
    
    else:
        return None


def _adjust_divide(nums: list, answer: float, scale: str):
    if len(nums) != 2:
        return None

    elif _adjust_scale(nums[0] / nums[1], answer, scale):
        return 1
    
    elif _adjust_scale(nums[1] / nums[0], answer, scale):
        return -1
    
    else:
        return None


def _adjust_ratio(nums: list, answer: float, scale: str):
    if len(nums) != 2:
        return None

    elif _adjust_scale((nums[0] - nums[1]) / nums[1], answer, scale):
        return 1
    
    elif _adjust_scale((nums[1] - nums[0]) / nums[0], answer, scale):
        return -1
    
    else:
        return None


def adjust_numbers(nums: list, answer: float, opr: str, scale: str):
    if opr == 'sum': return _adjust_sum(nums, answer, scale)
    elif opr == 'multiply': return _adjust_mul(nums, answer, scale)
    elif opr == 'average': return _adjust_avg(nums, answer, scale)
    elif opr == 'diff': return _adjust_diff(nums, answer, scale)
    elif opr == 'divide': return _adjust_divide(nums, answer, scale)
    elif opr == 'change-ratio': return _adjust_ratio(nums, answer, scale)
    return None
