# -*-coding:utf-8-*-
class FixedSizeList:
    """
    实现一个固定长度为3的列表，当列表满时，新元素会替换最旧的元素
    """

    def __init__(self):
        self.items = []
        self.max_size = 3

    def append(self, item):
        """
        添加新元素到列表中
        如果列表已满，会移除最旧的元素
        """
        if len(self.items) >= self.max_size:
            self.items.pop(0)  # 移除第一个(最旧的)元素
        self.items.append(item)

    def get_items(self):
        """
        获取当前列表中的所有元素
        """
        return self.items

    def __str__(self):
        return str(self.items)


# 使用示例
if __name__ == "__main__":
    fixed_list = FixedSizeList()
    hist_bev = fixed_list.items

    # 添加4个元素
    fixed_list.append(1)  # [1]
    fixed_list.append(2)  # [1, 2]
    fixed_list.append(3)  # [1, 2, 3]
    fixed_list.append(4)  # [2, 3, 4] - 1被移除

    print(fixed_list)  # 输出: [2, 3, 4]