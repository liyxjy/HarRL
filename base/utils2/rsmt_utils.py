import math
from turtle import st
import numpy as np
import torch
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from scipy import optimize as opt

class Node:
    '''
    @brief define structure of RC tree
    '''
    def __init__(self, name, resistance=0, capacitance=0, sink_capacitance=0, buffer_position=False):
        self.name = name                            # Node name
        self.resistance = resistance                # Resistance from parent to this node
        self.capacitance = capacitance              # Capacitance from parent to this node
        self.sink_capacitance = sink_capacitance    # Sink node capacitance
        self.children = []                          # List of child nodes
        self.buffer = False                         # Whether to insert a buffer
        self.buffer_cost = 0                        # Cost of inserting a buffer (delay reduction)
        self.opt_delay = 0                          # Optimized delay
        self.buffer_position = buffer_position      # Whether possibile position for buffer or not
        self.wl = 0                                 # wire length

    def add_child(self, child):
        self.children.append(child)


class BufferInsertion:
    '''
    @brief compute the elmore delay and decide whether insert buffer or not
    '''
    def __init__(self, root, buffer_delay, buffer_capacitance, buffer_resistance):
        self.root = root
        self.buffer_delay = buffer_delay
        self.buffer_capacitance = buffer_capacitance
        self.buffer_resistance = buffer_resistance

    def compute_capacitance(self, node):
        """
        @brief Recursively compute the total capacitance of the subtree,
         i.e., the downstream capacitance
        """
        if node is None:
            return 0
        if node.buffer:
            return self.buffer_capacitance
        if node.sink_capacitance>0 and node.name!=0:
            return node.sink_capacitance
        #if not node.children:
        #    return node.sink_capacitance
        cap = node.sink_capacitance + sum(child.capacitance for child in node.children) + \
             sum([self.compute_capacitance(child) for child in node.children])
        return cap
    
    def compute_elmore_delay(self, node):
        '''
        @brief compute elmore delay for edge which ends with node
        '''
        if node is None:
            return 0
        
        buffer = node.buffer
        node.buffer = False

        downstream_cap = self.compute_capacitance(node)

        if buffer:
            node_delay = self.buffer_delay + self.buffer_resistance * downstream_cap + node.resistance * (0.5 * node.capacitance + self.buffer_capacitance)
            node.buffer = True
        else:
            node_delay = node.resistance * downstream_cap + 0.5 * node.resistance * node.capacitance

        return node_delay
    
    
    def find_path(self, start, end, path=[]):
        '''
        @brief find path form start to end
        '''
        if not start:
            return None
        
        path = path + [start.name]

        if start.name == end.name:
            return path

        for child in start.children:
            new_path = self.find_path(child, end, path)
            if new_path:
                return new_path
    
        return None
           
    def calculate_depth(self, node, depth=0, depths=None):
        '''
        @brief calculate the depth of nodes in tree
        '''
        if depths is None:
            depths = {}

        depths[node.name] = depth

        for child in node.children:
            self.calculate_depth(child, depth + 1, depths)

        return depths
    
    def optimal_buffer_insertion(self, node_name_pair, depths, sink_num, N):
        '''
        @brief decide whether insert buffer or not by elmore delay
        '''
        if not node_name_pair:
            return 'No Node_name dictionary'
        
        if not depths:
            return 'No depths'
        
        node_order=sorted(depths.items(),key=lambda x:x[1],reverse=True)
        max_depth = node_order[0][1]

        b_num = 0
        for i in range(0, max_depth):
            for k, v in depths.items():
                current_node = node_name_pair[k]

                if v == max_depth-i and current_node.buffer_position == True:
                    
                    path = self.find_path(self.root, current_node)
                    buffer_parent = [j for j in path if j>=N]
                    sink_parent = [j for j in path if j<sink_num]
                    if len(buffer_parent)==1:
                        buffer_parent = sink_parent[-1]
                    else:
                        buffer_parent = buffer_parent[-2]
                    path_to_parent = path[path.index(buffer_parent) + 1 :]
                    

                    # elmore delay of node without buffer
                    delay_without_buffer = self.compute_elmore_delay(current_node)
                    for inner in range(len(path_to_parent) - 1):
                        delay_without_buffer += self.compute_elmore_delay(node_name_pair[path_to_parent[inner]])

                    # elmore delay of node with buffer
                    current_node.buffer = True
                    delay_with_buffer = self.compute_elmore_delay(current_node)
                    for inner in range(len(path_to_parent) - 1):
                        delay_with_buffer += self.compute_elmore_delay(node_name_pair[path_to_parent[inner]])

                    current_node.buffer_cost = delay_without_buffer - delay_with_buffer

                    if current_node.buffer_cost > 0:  # if buffer reduces delay then insert buffer
                        current_node.buffer = True
                        #current_node.opt_delay = delay_with_buffer
                        b_num += 1
                    else:
                        current_node.buffer = False
                        #current_node.opt_delay = delay_without_buffer
        #print(b_num)
        return b_num


class TimingTree:
    """
    Constructing the timing tree
    """
    def two_lines_inter(self, v_line, h_line):
        """
        @brief find the intersech node of two lines
        @param v_line the vertical line
        @param h_line the horizontal line
        @return a flag denoting whether intersecting, and intersect node
        """
        v_x1, v_y1, v_x2, v_y2 = v_line
        h_x1, h_y1, h_x2, h_y2 = h_line
        if (h_y1 >= v_y1) and (h_y1 <= v_y2) and \
            (v_x1 >= h_x1) and (v_x1 <= h_x2):
            return True, (v_x1, h_y1)
        else:
            return False, (-1, -1)

    def find_intersections(self, coords, connections):
        """
        @brief find all steiner points
        @param coords a dict whose key is 'id', and value is its coordinates
        @param connections a list composed of all res
        """
        steiner_points = set()
        for i in range(len(connections)):
            for j in range(i + 1, len(connections)):
                p1, q1 = connections[i][0], connections[i][1]
                p2, q2 = connections[j][0], connections[j][1]
                v_y_min1, v_y_max1 = min(coords[p1][1], coords[q1][1]), max(coords[p1][1], coords[q1][1])
                v_line_1 = (coords[p1][0], v_y_min1, coords[p1][0], v_y_max1)
                h_x_min1, h_x_max1 = min(coords[p1][0], coords[q1][0]), max(coords[p1][0], coords[q1][0])
                h_line_1 = (h_x_min1, coords[q1][1], h_x_max1, coords[q1][1])
                
                v_y_min2, v_y_max2 = min(coords[p2][1], coords[q2][1]), max(coords[p2][1], coords[q2][1])
                v_line_2 = (coords[p2][0], v_y_min2, coords[p2][0], v_y_max2)
                h_x_min2, h_x_max2 = min(coords[p2][0], coords[q2][0]), max(coords[p2][0], coords[q2][0])
                h_line_2 = (h_x_min2, coords[q2][1], h_x_max2, coords[q2][1])

                flag, inter_node = self.two_lines_inter(v_line_1, h_line_2)
                if flag:
                    steiner_points.add(inter_node)
        
                flag, inter_node = self.two_lines_inter(v_line_2, h_line_1)
                if flag:
                    steiner_points.add(inter_node)
        return steiner_points


    def insert_steiner(self, steiner_points, coordinates, pairs):
        """
        @brief insert steiner points to pairs
        @return {num:(,)} : a dictionary, key is the name of node (include steiner points), value is coordinate
        @return [(v, h, b)]  : a list, the connection pairs (include steiner points)
        steiner points number start with num(sink)+num(buffer position)
        """

        pair_with_steiner = []
        # add steiner points to coordinates
        n = len(coordinates)
        #i = 0
        for steiner in steiner_points:
            if steiner not in coordinates.values():
                coordinates[n] = steiner
                n += 1
                #i += 1

        #print("steiner_num", len(steiner_points), i)
        new_coords = {v : k for k, v in coordinates.items()}

        for pair in pairs:
            v, h, b = pair
            vX = coordinates[v][0]
            vY = coordinates[v][1]
            hX = coordinates[h][0]
            hY = coordinates[h][1]
            h_point = []
            v_point = []

            # find intermediate node
            for point in coordinates.values():
                sX, sY = point
                if sX == vX and min(vY, hY) <= sY <= max(vY, hY):
                    v_point.append(point)
                elif sY == hY and min(vX, hX) < sX < max(vX, hX):
                    h_point.append(point)
            if vY >= hY:
                v_point = sorted(v_point, key=lambda x:x[1], reverse=True)
            else:
                v_point = sorted(v_point, key=lambda x:x[1])
            if hX <= vX:
                h_point = sorted(h_point, key=lambda x:x[0], reverse=True)
            else:
                h_point = sorted(h_point, key=lambda x:x[0])

            # insert steiner point
            if v_point:
                pair_with_steiner.append([v, new_coords[v_point[0]], b])
                if len(v_point) > 1:
                    for i in range(1, len(v_point)):
                        pair_with_steiner.append([new_coords[v_point[i-1]], new_coords[v_point[i]], b])
                if h_point:
                    pair_with_steiner.append([new_coords[v_point[-1]], new_coords[h_point[0]], b])
                else:
                    pair_with_steiner.append([new_coords[v_point[-1]], h, b])

            if h_point:
                if not v_point:
                    pair_with_steiner.append([v, new_coords[h_point[0]], b])
                if len(h_point) > 1:
                    for i in range(1, len(h_point)):
                        pair_with_steiner.append([new_coords[h_point[i-1]], new_coords[h_point[i]], b])
                pair_with_steiner.append([new_coords[h_point[-1]], h, b])
            
            if not h_point and not v_point:
                pair_with_steiner.append([v, h, b])
            
            #pair_with_steiner = list(set(pair_with_steiner))
            pair_single = []
            for pair in pair_with_steiner:
                if pair[0]!=pair[1]:
                    pair_single.append(pair)

            pair_with_buffer_dict = defaultdict(list)
            for pair in pair_single:
                pair_with_buffer_dict[(int(pair[0]), int(pair[1]))].append(pair[2])
            pair_with_buffer = []
            for pair, b in pair_with_buffer_dict.items():
                pair_with_buffer.append(list(pair) + b)


            '''
            for pair in pair_single:
                v1, h1 = pair[0], pair[1]
                #length = math.dist(coords[v], coords[h])
                for other_pair in pair_single:
                    v2, h2, b2 = other_pair
                    if v1==v2 and h1==h2:
                        pair.append(b2)
                    pair_single.remove(other_pair)
                pair_with_buffer.append(pair)
            '''

        return coordinates, pair_with_buffer
    

    
    def find_buffer_positions(self, coords, connections, T):
        b_index = len(coords)
        connections_with_b = []
        #add_fun = lambda f1, f2 : lambda x : f1(x) + f2(x)
        
        for pair in connections:
            v, h = pair[0], pair[1]
            vX, vY = coords[v]
            hX, hY = coords[h]
            v_length = abs(vY - hY)
            h_length = abs(vX - hX)

            #path_num = len(pair) - 2
            #f1 = lambda x : -np.sin(2 * np.pi / T * x + pair[2])
            #if path_num>1:
            #    for i in range(path_num - 1):
            #        f2 = lambda x : -np.sin(2 * np.pi / T * x + pair[3 + i])
            #        f1 = add_fun(f1, f2)
            #max_position = opt.minimize(fun=f1, x0=T / 2, bounds=[(0, T)])
            #start_position = float(max_position.x)

            b = pair[2:]
            b_sin = b_cos = 0
            for i in b:
                b_sin += math.sin(i)
                b_cos += math.cos(i)
            start_position = ( -math.atan(b_sin / b_cos) / 2 / math.pi + 0.25 ) * T
            #print(start_position)
            #if start_position<0:
            #    start_position += T

            if (v_length + h_length) < start_position:
                connections_with_b.append((v, h))
                continue

            buffer_num = int((v_length + h_length - float(start_position)) // T + 1)
            #print(v_length + h_length, T)
            for i in range(buffer_num):
                buffer_position = float(start_position) + T * i
                if buffer_position<=v_length:
                    bX = vX
                    bY = vY - np.sign(vY - hY) * buffer_position
                else:
                    bX = vX - np.sign(vX - hX) * (buffer_position - v_length)
                    bY = hY
                
                coords[b_index] = (bX, bY)
                if i == 0 and buffer_num == 1:
                    connections_with_b.append((v, b_index))
                    connections_with_b.append((b_index, h))
                else:
                    if i == 0:
                        connections_with_b.append((v, b_index))
                    elif i == buffer_num - 1:
                        connections_with_b.append((b_index - 1, b_index))
                        connections_with_b.append((b_index, h))
                    else:
                        connections_with_b.append((b_index - 1, b_index))
                    
                b_index += 1

        return coords, connections_with_b
    

    def build_tree(self, pair_with_steiner, coords_with_steiner, sink_cap, wire_capacitance, wire_resistance, N):
        """
        @brief build tree for elmore delay
        @param pair_with_steiner [(,)]: a list, connections of nodes
        @param coords_with_steiner {num:(,)}: a dictionary, key is the name of node, value is coordinates
        @param sink_cap: a dictionary, key is the name of sink, value is the capacitance of sink
        @param wire_capasitance, wire_resistance: float
        @param N: num of sinks and steiner points
        @return root: a Node, source
        @return nodes: a dictionary, key is the name of node, value is Node
        """
        visited = [False] * len(coords_with_steiner)
        queue = deque([0])
        visited[0] = True
        root = Node(name=0)
        nodes = {0: root}

        while queue:
            current = queue.popleft()  # 取出队列的第一个元素
            current_tree_node = nodes[current]  # 获取当前树节点

            for pair in pair_with_steiner:
                v, h = pair
                vX, vY = coords_with_steiner[v]
                hX, hY = coords_with_steiner[h]
                v_length = abs(vY - hY)
                h_length = abs(vX - hX)
                #v_location = coords_with_steiner[v]
                #h_location = coords_with_steiner[h]
                sink_capacitance = 0
                buffer_position = False
                if v==current and visited[h]==False:
                    visited[h] = True  # 标记为已访问
                    queue.append(h)  # 加入队列
                    wire_length = v_length + h_length
                    #print("wire length", (v,h), wire_length)
                    point_capacitance = wire_capacitance * wire_length
                    point_resistance = wire_resistance * wire_length
                    if h<len(sink_cap):
                        sink_capacitance = sink_cap[h]
                    elif h>=N:
                        buffer_position = True
                    child_node = Node(name = h, resistance = point_resistance, capacitance = point_capacitance, sink_capacitance=sink_capacitance, buffer_position=buffer_position)
                    child_node.wl = wire_length
                    current_tree_node.children.append(child_node)  # 添加子节点到当前树节点
                    nodes[h] = child_node  # 记录新节点
                elif h==current and visited[v]==False:
                    visited[v] = True  # 标记为已访问
                    queue.append(v)  # 加入队列
                    wire_length = v_length + h_length
                    #print("wire length", (h,v), wire_length)
                    point_capacitance = wire_capacitance * wire_length
                    point_resistance = wire_resistance * wire_length
                    if v<len(sink_cap):
                        sink_capacitance = sink_cap[v]
                    elif v>=N:
                        buffer_position = True
                    child_node = Node(name = v, resistance = point_resistance, capacitance = point_capacitance, sink_capacitance=sink_capacitance, buffer_position=buffer_position)
                    child_node.wl = wire_length
                    current_tree_node.children.append(child_node)  # 添加子节点到当前树节点
                    nodes[v] = child_node  # 记录新节点
        return root, nodes



class Evaluator:
    """
    @brief evaluator engine
    receive a batch of coordinates and res, returns the elmore delay after buffer insertion
    """
    def __init__(self, degree):
        self.wire_resistance = 0.003000
        self.wire_capacitance = 2.000000e-15
        if degree<=10:
            self.buffer_delay = 3e-21
            self.buffer_capacitance = 5e-16
            self.buffer_resistance = 0.0002
        elif degree<=20:
            self.buffer_delay = 3e-21
            self.buffer_capacitance = 4e-16
            self.buffer_resistance = 0.00015
        else:
            self.buffer_delay = 3e-21
            self.buffer_capacitance = 3e-16
            self.buffer_resistance = 0.00012
        self.tree = TimingTree()

    def eval_batch(self, input_batch, output_batch, degree):
        lengths = []
        slacks = []
        skews = []
        nums = []
        batch_size = input_batch.shape[0]
        for i in range(batch_size):
            coords = {j: tuple(input_batch[i, j, :2].tolist()) for j in range(degree)}
            sink_cap = {j: input_batch[i, j, 2].item() / 1e15 for j in range(degree)}
            connections = [(output_batch[i, 3*k].item(), output_batch[i, 3*k+1].item(), output_batch[i, 3*k+2].item()) for k in range(degree - 1)]
            slack, skew, length, b_num = self.eval_func(coords, sink_cap, connections)
            lengths.append(length)
            slacks.append(slack)
            skews.append(skew)
            nums.append(b_num)
        # return np.array(slacks) * 1e18, np.array(skews) * 1e18, np.array(lengths), np.array(nums)
        delay = np.array(slacks) * 1e18
        skew = np.array(skews) * 1e18
        lengths = np.array(lengths)
        buffer_num = np.array(nums)
        return 0.4 * delay + 0.4 * skew + 0.2 * lengths + 0.2 * buffer_num
    
    def eval_func(self, coords, sink_cap, connections):
        """
        @brief get the total elmore delay of all sinks
        @param coords a dict mapping one sink to its coordinates
        @param sink_cap a dict mapping one sink to its capacitance
        @param wire_resistance unit wire resistance
        @param wire_capacitance unit wire capacitance
        @param connections the routing result represented by res, and buffer_positions possibile positions for buffer insertion
        """       
        T = math.sqrt(2 * self.buffer_capacitance * self.buffer_resistance / self.wire_resistance / self.wire_capacitance)
        #coords_with_b, pairs_with_b = self.tree.find_buffer_positions(coords, connections)
        steiner_points = self.tree.find_intersections(coords, connections)
        coords_with_s, pairs_with_s = self.tree.insert_steiner(steiner_points, coords, connections)
        num_with_steiner = len(coords_with_s)
        coords_with_b, pairs_with_b = self.tree.find_buffer_positions(coords_with_s, pairs_with_s, T)
        num_buffer = len(coords_with_b) - len(sink_cap)
        #print('buffer candidates:', num_buffer)
        root, nodes = self.tree.build_tree(pairs_with_b, coords_with_b, sink_cap, self.wire_capacitance, self.wire_resistance, num_with_steiner)
        buffer = BufferInsertion(root, self.buffer_delay, self.buffer_capacitance, self.buffer_resistance)

        depth = buffer.calculate_depth(root)
        sink_num = len(sink_cap)
        #buffer.optimal_buffer_insertion(nodes, depth, sink_num, num_with_steiner)
        b_num = buffer.optimal_buffer_insertion(nodes, depth, sink_num, num_with_steiner)
        #self.plot_buffer(coords_with_b, connections, nodes, num_with_steiner)
        
        sink_delay = [0] * (sink_num-1)
        wirelength = 0 

        for i in range(len(nodes)):
            nodes[i].opt_delay = buffer.compute_elmore_delay(nodes[i])
            wirelength += nodes[i].wl
        
        for i in range(1,sink_num):
            sink = nodes[i]
            path = buffer.find_path(root, sink)
            for inner in range(1, len(path)):
                sink_delay[i-1] += nodes[path[inner]].opt_delay
        delayMax = max(sink_delay)
        delayMin = min(sink_delay)

        return delayMax, delayMax - delayMin, wirelength, b_num
    
    def plot_buffer(self, coords, connection, nodes, N):
        """
        @ param connection: a dict, initial connection (i.e. without steiner, buffer position)
        @ param coords: a dict, coordinates with buffer position
        @ param nodes: a dict, name node pair
        """
        for pair in connection:
            v = coords[pair[0]]
            h = coords[pair[1]]
            plt.plot([v[0], v[0]], [v[1], h[1]], 'k')
            plt.plot([v[0], h[0]], [h[1], h[1]], 'k')

        for i in range(len(connection)+1, len(coords)):
            node = nodes[i]
            if node.buffer==True:
                plt.plot(coords[i][0], coords[i][1], '^:r')
            elif i>=N:
                plt.plot(coords[i][0], coords[i][1], '^:b')
        
        plt.plot(coords[0][0], coords[0][1], 's:y')
        for i in range(1, len(connection)+1):
            plt.plot(coords[i][0], coords[i][1], 'o:y')
        
        ax = plt.subplot()
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig('e10.png')
