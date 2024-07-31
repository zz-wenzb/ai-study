import numpy as np
from 神经网络.utils.features import prepare_for_training
from 神经网络.utils.hypothesis import sigmoid, sigmoid_gradient


class MultilayerPerceptron:
    def __init__(self, data, labels, layers, normalize_data=False):
        data_processed = prepare_for_training(data, normalize_data=normalize_data)[0]
        self.data = data_processed
        self.labels = labels
        self.layers = layers  # 784 25 10
        self.normalize_data = normalize_data
        self.thetas = MultilayerPerceptron.thetas_init(layers)

    def predict(self, data):
        """
        使用训练好的模型进行预测。

        参数:
        - data: numpy数组，输入数据，用于进行预测。

        返回:
        - 预测结果，numpy数组，形状为(num_examples, 1)，其中num_examples是数据样本数量。
        """
        # 准备数据，进行必要的预处理，如归一化，并仅保留处理后的数据
        data_processed = prepare_for_training(data, normalize_data=self.normalize_data)[0]
        # 获取处理后的数据样本数量
        num_examples = data_processed.shape[0]

        # 使用多层感知器模型进行前向传播，获得预测结果
        predictions = MultilayerPerceptron.feedforward_propagation(data_processed, self.thetas, self.layers)

        # 返回预测结果中每行最大值的索引，即预测的类别，并重塑为(num_examples, 1)的形状
        return np.argmax(predictions, axis=1).reshape((num_examples, 1))

    def train(self, max_iterations=1000, alpha=0.1):
        """
        训练多层感知机模型。

        通过梯度下降法优化网络的权重参数，以最小化成本函数。

        参数:
        - max_iterations: int，最大迭代次数，默认为1000。
        - alpha: float，学习率，默认为0.1。

        返回:
        - thetas: list，优化后的权重参数列表。
        - cost_history: list，每次迭代的成本记录。
        """

        # 将权重参数列表展平为一维数组，以便进行优化处理
        unrolled_theta = MultilayerPerceptron.thetas_unroll(self.thetas)

        # 使用梯度下降法优化展平后的权重参数
        # 这里是核心优化步骤，通过迭代更新权重以最小化成本函数
        (optimized_theta, cost_history) = MultilayerPerceptron.gradient_descent(self.data, self.labels, unrolled_theta,
                                                                                self.layers, max_iterations, alpha)

        # 将优化后的展平权重参数恢复为原始的列表结构
        self.thetas = MultilayerPerceptron.thetas_roll(optimized_theta, self.layers)

        # 返回优化后的权重参数和成本历史记录
        return self.thetas, cost_history


    @staticmethod
    def thetas_init(layers):
        num_layers = len(layers)
        thetas = {}
        for layer_index in range(num_layers - 1):
            """
                            会执行两次，得到两组参数矩阵：25*785 , 10*26
            """
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]
            # 这里需要考虑到偏置项，记住一点偏置的个数跟输出的结果是一致的
            thetas[layer_index] = np.random.rand(out_count, in_count + 1) * 0.05  # 随机进行初始化操作，值尽量小一点
        return thetas

    @staticmethod
    def thetas_unroll(thetas):
        """
        将多层theta参数矩阵展平并连接成一个一维数组。

        参数:
        thetas: 一个包含多层theta参数矩阵的列表，每层theta是一个二维数组。

        返回值:
        一个展平后的一维数组，包含了所有theta参数的值。
        """
        # 初始化一个空的一维数组，用于存储展平后的theta参数
        num_theta_layers = len(thetas)
        unrolled_theta = np.array([])
        # 遍历每一层theta参数矩阵
        for theta_layer_index in range(num_theta_layers):
            # 将当前层的theta参数矩阵展平，并追加到unrolled_theta数组中
            # hstack操作可以将多个数组水平拼接成一个数组
            # flatten操作将二维数组展平为一维数组
            """
            hstack :
            [[1, 2], [3, 4]] ,[[5, 6], [7, 8]]
            ==>
            ([[1, 2, 5, 6],
            [3, 4, 7, 8]]
            
            flatten:
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            ==> [1, 2, 3, 4, 5, 6, 7, 8, 9]
            """
            unrolled_theta = np.hstack((unrolled_theta, thetas[theta_layer_index].flatten()))
        # 返回展平后的一维数组
        return unrolled_theta

    @staticmethod
    def gradient_descent(data, labels, unrolled_theta, layers, max_iterations, alpha):
        """
        使用梯度下降法优化神经网络的权重参数。

        参数：
        data: 输入数据，用于训练神经网络。
        labels: 输入数据的标签，用于计算误差。
        unrolled_theta: 初始的展平权重参数，表示神经网络的初始权重状态。
        layers: 神经网络的层数结构，用于确定权重参数的形状。
        max_iterations: 最大迭代次数，用于控制梯度下降的迭代轮数。
        alpha: 学习率，控制权重更新的速度。

        返回：
        optimized_theta: 优化后的展平权重参数，表示经过训练后的神经网络权重。
        cost_history: 成本历史记录，包含每次迭代的成本值。
        """
        # 初始化优化后的权重参数为初始权重参数
        optimized_theta = unrolled_theta
        # 初始化成本历史记录为空列表，用于存储每次迭代的成本值
        cost_history = []

        # 遍历最大迭代次数，执行梯度下降更新权重参数
        for _ in range(max_iterations):
            # 计算当前权重参数下的成本值
            cost = MultilayerPerceptron.cost_function(data, labels,
                                                      MultilayerPerceptron.thetas_roll(optimized_theta, layers), layers)
            # 将成本值添加到历史记录中
            cost_history.append(cost)
            # 计算权重参数的梯度
            theta_gradient = MultilayerPerceptron.gradient_step(data, labels, optimized_theta, layers)
            # 更新权重参数
            optimized_theta = optimized_theta - alpha * theta_gradient
        # 返回优化后的权重参数和成本历史记录
        return optimized_theta, cost_history

    # 计算梯度
    # 参数:
    #   data: 输入数据
    #   labels: 数据对应的标签
    #   optimized_theta: 当前优化的权重参数
    #   layers: 网络结构的层数
    # 返回:
    #   thetas_unrolled_gradients: 展平的梯度矩阵
    @staticmethod
    def gradient_step(data, labels, optimized_theta, layers):
        theta = MultilayerPerceptron.thetas_roll(optimized_theta, layers)
        thetas_rolled_gradients = MultilayerPerceptron.back_propagation(data, labels, theta, layers)
        thetas_unrolled_gradients = MultilayerPerceptron.thetas_unroll(thetas_rolled_gradients)
        return thetas_unrolled_gradients

    # 反向传播算法
    # 参数:
    #   data: 输入数据
    #   labels: 数据对应的标签
    #   thetas: 展开的权重参数
    #   layers: 网络结构的层数
    # 返回:
    #   deltas: 每一层的梯度
    @staticmethod
    def back_propagation(data, labels, thetas, layers):
        num_layers = len(layers)
        (num_examples, num_features) = data.shape
        num_label_types = layers[-1]

        deltas = {}  # 存储每层的梯度
        # 初始化操作
        for layer_index in range(num_layers - 1):
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]
            deltas[layer_index] = np.zeros((out_count, in_count + 1))  # 25*785 10*26
        for example_index in range(num_examples):
            layers_inputs = {}
            layers_activations = {}
            layers_activation = data[example_index, :].reshape((num_features, 1))  # 785*1
            layers_activations[0] = layers_activation
            # 逐层计算
            for layer_index in range(num_layers - 1):
                layer_theta = thetas[layer_index]  # 得到当前权重参数值 25*785   10*26
                layer_input = np.dot(layer_theta, layers_activation)  # 第一次得到25*1 第二次10*1
                layers_activation = np.vstack((np.array([[1]]), sigmoid(layer_input)))
                layers_inputs[layer_index + 1] = layer_input  # 后一层计算结果
                layers_activations[layer_index + 1] = layers_activation  # 后一层经过激活函数后的结果
            output_layer_activation = layers_activation[1:, :]

            delta = {}
            # 标签处理
            bitwise_label = np.zeros((num_label_types, 1))
            bitwise_label[labels[example_index][0]] = 1
            # 计算输出层和真实值之间的差异
            delta[num_layers - 1] = output_layer_activation - bitwise_label

            # 遍历循环 L L-1 L-2 ...2
            for layer_index in range(num_layers - 2, 0, -1):
                layer_theta = thetas[layer_index]
                next_delta = delta[layer_index + 1]
                layer_input = layers_inputs[layer_index]
                layer_input = np.vstack((np.array((1)), layer_input))
                # 按照公式进行计算
                delta[layer_index] = np.dot(layer_theta.T, next_delta) * sigmoid_gradient(layer_input)
                # 过滤掉偏置参数
                delta[layer_index] = delta[layer_index][1:, :]
            for layer_index in range(num_layers - 1):
                layer_delta = np.dot(delta[layer_index + 1], layers_activations[layer_index].T)
                deltas[layer_index] = deltas[layer_index] + layer_delta  # 第一次25*785  第二次10*26

        for layer_index in range(num_layers - 1):
            deltas[layer_index] = deltas[layer_index] * (1 / num_examples)

        return deltas

    # 计算成本函数
    # 参数:
    #   data: 输入数据
    #   labels: 数据对应的标签
    #   thetas: 展开的权重参数
    #   layers: 网络结构的层数
    # 返回:
    #   cost: 成本函数的值
    @staticmethod
    def cost_function(data, labels, thetas, layers):
        num_layers = len(layers)
        num_examples = data.shape[0]
        num_labels = layers[-1]

        # 前向传播走一次
        predictions = MultilayerPerceptron.feedforward_propagation(data, thetas, layers)
        # 制作标签，每一个样本的标签都得是one-hot
        bitwise_labels = np.zeros((num_examples, num_labels))
        for example_index in range(num_examples):
            bitwise_labels[example_index][labels[example_index][0]] = 1
        bit_set_cost = np.sum(np.log(predictions[bitwise_labels == 1]))
        bit_not_set_cost = np.sum(np.log(1 - predictions[bitwise_labels == 0]))
        cost = (-1 / num_examples) * (bit_set_cost + bit_not_set_cost)
        return cost

    # 前向传播
    # 参数:
    #   data: 输入数据
    #   thetas: 展开的权重参数
    #   layers: 网络结构的层数
    # 返回:
    #   输出层结果
    @staticmethod
    def feedforward_propagation(data, thetas, layers):
        num_layers = len(layers)
        num_examples = data.shape[0]
        in_layer_activation = data

        # 逐层计算
        for layer_index in range(num_layers - 1):
            theta = thetas[layer_index]
            out_layer_activation = sigmoid(np.dot(in_layer_activation, theta.T))
            # 正常计算完之后是num_examples*25,但是要考虑偏置项 变成num_examples*26
            out_layer_activation = np.hstack((np.ones((num_examples, 1)), out_layer_activation))
            in_layer_activation = out_layer_activation

        # 返回输出层结果,结果中不要偏置项了
        return in_layer_activation[:, 1:]
    @staticmethod
    def thetas_roll(unrolled_thetas, layers):
        """
        将展开的theta参数重新滚动为每层的权重矩阵。

        参数：
        unrolled_thetas：一维数组，包含了所有层的theta参数。
        layers：列表，指定了神经网络每层的单元数（包括输入层和输出层）。

        返回：
        字典，包含了每层的权重矩阵。
        """
        # 获取神经网络的层数
        num_layers = len(layers)
        # 初始化一个空字典，用于存储每层的权重矩阵
        thetas = {}
        # 初始化展开参数的偏移量
        unrolled_shift = 0
        # 遍历除最后一层之外的每一层
        for layer_index in range(num_layers - 1):
            # 获取当前层和下一层的单元数
            in_count = layers[layer_index]
            out_count = layers[layer_index + 1]

            # 计算当前层权重矩阵的宽度（包含偏置项）
            thetas_width = in_count + 1
            # 计算当前层权重矩阵的高度
            thetas_height = out_count
            # 计算当前层权重矩阵的体积（即元素总数）
            thetas_volume = thetas_width * thetas_height
            # 获取当前层权重矩阵在展开参数中的起始索引
            start_index = unrolled_shift
            # 获取当前层权重矩阵在展开参数中的结束索引
            end_index = unrolled_shift + thetas_volume
            # 提取当前层的权重参数
            layer_theta_unrolled = unrolled_thetas[start_index:end_index]
            # 将提取的权重参数重塑为对应的权重矩阵，并存储到字典中
            thetas[layer_index] = layer_theta_unrolled.reshape((thetas_height, thetas_width))
            # 更新展开参数的偏移量
            unrolled_shift = unrolled_shift + thetas_volume

        # 返回包含每层权重矩阵的字典
        return thetas
