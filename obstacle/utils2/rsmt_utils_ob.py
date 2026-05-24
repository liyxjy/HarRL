import math
import numpy as np
import torch
from collections import deque, defaultdict
import matplotlib.pyplot as plt
from scipy import optimize as opt


class Node:
    def __init__(self, name, resistance=0, capacitance=0, sink_capacitance=0, buffer_position=False):
        self.name = name
        self.resistance = resistance
        self.capacitance = capacitance
        self.sink_capacitance = sink_capacitance
        self.children = []
        self.buffer = False
        self.buffer_cost = 0
        self.opt_delay = 0
        self.buffer_position = buffer_position
        self.wl = 0

    def add_child(self, child):
        self.children.append(child)


class BufferInsertion:
    def __init__(self, root, buffer_delay, buffer_capacitance, buffer_resistance):
        self.root = root
        self.buffer_delay = buffer_delay
        self.buffer_capacitance = buffer_capacitance
        self.buffer_resistance = buffer_resistance

    def compute_capacitance(self, node):
        if node is None:
            return 0
        if node.buffer:
            return self.buffer_capacitance
        if node.sink_capacitance > 0 and node.name != 0:
            return node.sink_capacitance
        cap = node.sink_capacitance + sum(child.capacitance for child in node.children) + \
              sum([self.compute_capacitance(child) for child in node.children])
        return cap

    def compute_elmore_delay(self, node):
        if node is None:
            return 0
        buffer = node.buffer
        node.buffer = False
        downstream_cap = self.compute_capacitance(node)
        if buffer:
            node_delay = self.buffer_delay + self.buffer_resistance * downstream_cap + \
                         node.resistance * (0.5 * node.capacitance + self.buffer_capacitance)
            node.buffer = True
        else:
            node_delay = node.resistance * downstream_cap + 0.5 * node.resistance * node.capacitance
        return node_delay

    def find_path(self, start, end, path=[]):
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
        if depths is None:
            depths = {}
        depths[node.name] = depth
        for child in node.children:
            self.calculate_depth(child, depth + 1, depths)
        return depths

    def optimal_buffer_insertion(self, node_name_pair, depths, sink_num, N):
        if not node_name_pair:
            return 'No Node_name dictionary'
        if not depths:
            return 'No depths'
        node_order = sorted(depths.items(), key=lambda x: x[1], reverse=True)
        max_depth = node_order[0][1]
        b_num = 0
        for i in range(0, max_depth):
            for k, v in depths.items():
                current_node = node_name_pair[k]
                if v == max_depth - i and current_node.buffer_position == True:
                    path = self.find_path(self.root, current_node)
                    buffer_parent = [j for j in path if j >= N]
                    sink_parent = [j for j in path if j < sink_num]
                    if len(buffer_parent) == 1:
                        buffer_parent = sink_parent[-1]
                    else:
                        buffer_parent = buffer_parent[-2]
                    path_to_parent = path[path.index(buffer_parent) + 1:]
                    delay_without_buffer = self.compute_elmore_delay(current_node)
                    for inner in range(len(path_to_parent) - 1):
                        delay_without_buffer += self.compute_elmore_delay(node_name_pair[path_to_parent[inner]])
                    current_node.buffer = True
                    delay_with_buffer = self.compute_elmore_delay(current_node)
                    for inner in range(len(path_to_parent) - 1):
                        delay_with_buffer += self.compute_elmore_delay(node_name_pair[path_to_parent[inner]])
                    current_node.buffer_cost = delay_without_buffer - delay_with_buffer
                    if current_node.buffer_cost > 0:
                        current_node.buffer = True
                        b_num += 1
                    else:
                        current_node.buffer = False


class TimingTree:
    def two_lines_inter(self, v_line, h_line):
        v_x1, v_y1, v_x2, v_y2 = v_line
        h_x1, h_y1, h_x2, h_y2 = h_line
        if (h_y1 >= v_y1) and (h_y1 <= v_y2) and \
                (v_x1 >= h_x1) and (v_x1 <= h_x2):
            return True, (v_x1, h_y1)
        else:
            return False, (-1, -1)

    def find_intersections(self, coords, connections):
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
        pair_with_steiner = []
        n = len(coordinates)
        for steiner in steiner_points:
            if steiner not in coordinates.values():
                coordinates[n] = steiner
                n += 1
        new_coords = {v: k for k, v in coordinates.items()}
        for pair in pairs:
            v, h, b = pair
            vX = coordinates[v][0]
            vY = coordinates[v][1]
            hX = coordinates[h][0]
            hY = coordinates[h][1]
            h_point = []
            v_point = []
            for point in coordinates.values():
                sX, sY = point
                if sX == vX and min(vY, hY) <= sY <= max(vY, hY):
                    v_point.append(point)
                elif sY == hY and min(vX, hX) < sX < max(vX, hX):
                    h_point.append(point)
            if vY >= hY:
                v_point = sorted(v_point, key=lambda x: x[1], reverse=True)
            else:
                v_point = sorted(v_point, key=lambda x: x[1])
            if hX <= vX:
                h_point = sorted(h_point, key=lambda x: x[0], reverse=True)
            else:
                h_point = sorted(h_point, key=lambda x: x[0])
            if v_point:
                pair_with_steiner.append([v, new_coords[v_point[0]], b])
                if len(v_point) > 1:
                    for i in range(1, len(v_point)):
                        pair_with_steiner.append([new_coords[v_point[i - 1]], new_coords[v_point[i]], b])
                if h_point:
                    pair_with_steiner.append([new_coords[v_point[-1]], new_coords[h_point[0]], b])
                else:
                    pair_with_steiner.append([new_coords[v_point[-1]], h, b])
            if h_point:
                if not v_point:
                    pair_with_steiner.append([v, new_coords[h_point[0]], b])
                if len(h_point) > 1:
                    for i in range(1, len(h_point)):
                        pair_with_steiner.append([new_coords[h_point[i - 1]], new_coords[h_point[i]], b])
                pair_with_steiner.append([new_coords[h_point[-1]], h, b])
            if not h_point and not v_point:
                pair_with_steiner.append([v, h, b])
            pair_single = []
            for pair in pair_with_steiner:
                if pair[0] != pair[1]:
                    pair_single.append(pair)
            pair_with_buffer_dict = defaultdict(list)
            for pair in pair_single:
                pair_with_buffer_dict[(int(pair[0]), int(pair[1]))].append(pair[2])
            pair_with_buffer = []
            for pair, b in pair_with_buffer_dict.items():
                pair_with_buffer.append(list(pair) + b)
        return coordinates, pair_with_buffer

    def find_buffer_positions(self, coords, connections, T):
        b_index = len(coords)
        connections_with_b = []
        for pair in connections:
            v, h = pair[0], pair[1]
            vX, vY = coords[v]
            hX, hY = coords[h]
            v_length = abs(vY - hY)
            h_length = abs(vX - hX)
            b = pair[2:]
            b_sin = b_cos = 0
            for i in b:
                b_sin += math.sin(i)
                b_cos += math.cos(i)
            if abs(b_cos) < 1e-12:
                b_cos = 1e-12
            start_position = (-math.atan(b_sin / b_cos) / 2 / math.pi + 0.25) * T
            if (v_length + h_length) < start_position:
                connections_with_b.append((v, h))
                continue
            buffer_num = int((v_length + h_length - float(start_position)) // T + 1)
            for i in range(buffer_num):
                buffer_position = float(start_position) + T * i
                if buffer_position <= v_length:
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
        visited = [False] * len(coords_with_steiner)
        queue = deque([0])
        visited[0] = True
        root = Node(name=0)
        nodes = {0: root}
        while queue:
            current = queue.popleft()
            current_tree_node = nodes[current]
            for pair in pair_with_steiner:
                v, h = pair
                vX, vY = coords_with_steiner[v]
                hX, hY = coords_with_steiner[h]
                v_length = abs(vY - hY)
                h_length = abs(vX - hX)
                sink_capacitance = 0
                buffer_position = False
                if v == current and visited[h] == False:
                    visited[h] = True
                    queue.append(h)
                    wire_length = v_length + h_length
                    point_capacitance = wire_capacitance * wire_length
                    point_resistance = wire_resistance * wire_length
                    if h < len(sink_cap):
                        sink_capacitance = sink_cap[h]
                    elif h >= N:
                        buffer_position = True
                    child_node = Node(name=h, resistance=point_resistance, capacitance=point_capacitance,
                                      sink_capacitance=sink_capacitance, buffer_position=buffer_position)
                    child_node.wl = wire_length
                    current_tree_node.children.append(child_node)
                    nodes[h] = child_node
                elif h == current and visited[v] == False:
                    visited[v] = True
                    queue.append(v)
                    wire_length = v_length + h_length
                    point_capacitance = wire_capacitance * wire_length
                    point_resistance = wire_resistance * wire_length
                    if v < len(sink_cap):
                        sink_capacitance = sink_cap[v]
                    elif v >= N:
                        buffer_position = True
                    child_node = Node(name=v, resistance=point_resistance, capacitance=point_capacitance,
                                      sink_capacitance=sink_capacitance, buffer_position=buffer_position)
                    child_node.wl = wire_length
                    current_tree_node.children.append(child_node)
                    nodes[v] = child_node
        return root, nodes


# ══════════════════════════════════════════════════════════════════════
#  ObstacleTree  —  在原有基础上增加：
#    1. insert_obstacle_coords : 把障碍物四顶点加入 coords
#    2. reroute_around_obstacles : 做几何绕障
# ══════════════════════════════════════════════════════════════════════
class ObstacleTree:
    def __init__(self):
        self.obstacle = None
        self.ob_coord_base = None   # 障碍物顶点在 coords 中的起始编号

    def input_ob(self, obs):
        self.obstacle = obs
        self.ob_coord_base = None
        
    def check_horizontal_vertical_intersection_strict(self, horizontal_line, vertical_line):
    
        h_x1, h_y1, h_x2, h_y2 = horizontal_line 
        v_x1, v_y1, v_x2, v_y2 = vertical_line
        
        h_x_min, h_x_max = min(h_x1, h_x2), max(h_x1, h_x2)
        v_y_min, v_y_max = min(v_y1, v_y2), max(v_y1, v_y2)
        
        if (h_x_min < v_x1 < h_x_max) and (v_y_min < h_y1 < v_y_max):
            return True, (v_x1, h_y1)
        else:
            #print(horizontal_line, vertical_line, False)
            return False, None

    def detect_intersections(self, v, h):
        intersect_ob = []
        vX, vY = v
        hX, hY = h
        wire_l = abs(vX - hX) + abs(vY - hY)
        v_line = (vX, vY, vX, hY)
        h_line = (hX, hY, vX, hY)
        #print(v_line, h_line)
                
        for i in range(len(self.obstacle)):
            ob = self.obstacle[i]
            x1, y1, x2, y2 = ob
            h1 = (x1, y1, x2, y1)
            h2 = (x1, y2, x2, y2)
            v1 = (x1, y1, x1, y2)
            v2 = (x2, y1, x2, y2)
            
            if vY != hY and vX != hX:
                flag1, p1 = self.check_horizontal_vertical_intersection_strict(h1, v_line)
                flag2, p2 = self.check_horizontal_vertical_intersection_strict(h2, v_line)
                if flag1 and flag2:
                    p = min(abs(x1 - vX) / wire_l, abs(x2 - vX) / wire_l)
                    ob_l = min(abs(x1 - vX), abs(x2 - vX))
                    intersect_ob.append((i, p, 2*ob_l)) 
                elif flag1 or flag2:
                        #if hY != y1 and hY != y2:
                    p = min(abs(x1 - vX) / wire_l, abs(x2 - vX) / wire_l, abs(y1 - hY) / wire_l, abs(y2 - hY))
                    intersect_ob.append((i, p, 0.0))
                else:
                    flag3, p3 = self.check_horizontal_vertical_intersection_strict(h_line, v1)
                    flag4, p4 = self.check_horizontal_vertical_intersection_strict(h_line, v2)
                    if vX != x1 and vX != x2:
                        if flag3 and flag4:
                            p = min(abs(y1 - hY) / wire_l, abs(y2 - hY) / wire_l)
                            ob_l = min(abs(y1 - hY), abs(y2 - hY))
                            intersect_ob.append((i, p, 2*ob_l))
                    else:
                        if flag3 or flag4:
                            p = min(abs(y1 - hY) / wire_l, abs(y2 - hY) / wire_l)
                            intersect_ob.append((i, p, 0.0))
            elif vY == hY:
                flag3, p3 = self.check_horizontal_vertical_intersection_strict(h_line, v1)
                flag4, p4 = self.check_horizontal_vertical_intersection_strict(h_line, v2)
                if flag3 and flag4:
                    p = min(abs(y1 - hY) / wire_l, abs(y2 - hY) / wire_l)
                    ob_l = min(abs(y1 - hY), abs(y2 - hY))
                    intersect_ob.append((i, p, 2*ob_l))
            elif vX == hX:
                flag1, p1 = self.check_horizontal_vertical_intersection_strict(h1, v_line)
                flag2, p2 = self.check_horizontal_vertical_intersection_strict(h2, v_line)
                if flag1 and flag2:
                    p = min(abs(x1 - vX) / wire_l, abs(x2 - vX) / wire_l)
                    ob_l = min(abs(x1 - vX), abs(x2 - vX))
                    intersect_ob.append((i, p, 2*ob_l))
        return intersect_ob
    
    def change_sin(self, coords, connections, obstacle_sin, T):
        connections_with_ob = []
        for i, pair in enumerate(connections):
            obstacle_sin_pair = obstacle_sin[i]
            v = coords[pair[0]]
            h = coords[pair[1]]
            b = pair[2]
            vX, vY = v
            hX, hY = h
            l = abs(vX - hX) + abs(vY - hY) 
            intersect_ob = self.detect_intersections(v, h)
            if intersect_ob:
                ob_b = []
                for ob in intersect_ob:
                    i, p, ob_l = ob
                    ob_b.append([obstacle_sin_pair[i], p, ob_l])
                    b += obstacle_sin_pair[i]
                sum_ob_l = 0.0
                for cc in ob_b:
                    sum_ob_l += cc[2]
                connections_with_ob.append([pair[0], pair[1], b % T, sum_ob_l / l])
            else:
                connections_with_ob.append([pair[0], pair[1], b, 0.0])
        return coords, connections_with_ob 

    def insert_obstacle_coords(self, coords):
        self.ob_coord_base = len(coords)
        return coords, self.ob_coord_base
 
    def _get_or_create_coord(self, coords, pt):
        for k, v in coords.items():
            if abs(v[0] - pt[0]) < 1e-9 and abs(v[1] - pt[1]) < 1e-9:
                return k
        new_id = len(coords)
        coords[new_id] = pt
        return new_id
 
    def _seg_cross_obstacle(self, ax, ay, bx, by, x1, y1, x2, y2):
        
        ox1, ox2 = min(x1, x2), max(x1, x2)
        oy1, oy2 = min(y1, y2), max(y1, y2)

        def in_obstacle(px, py):
            return ox1 < px < ox2 and oy1 < py < oy2

        if ax == bx:
            sx = ax
            sy1, sy2 = min(ay, by), max(ay, by)
            if ox1 < sx < ox2 and sy1 < oy2 and sy2 > oy1:
                if in_obstacle(bx, by):
                    enter_pt = (sx, oy1) if ay <= by else (sx, oy2)
                    return 2, enter_pt, (bx, by)
                
                if in_obstacle(ax, ay):
                    exit_pt = (sx, oy2) if ay <= by else (sx, oy1)
                    return 3, (ax, ay), exit_pt
                
                enter_y = max(sy1, oy1)
                exit_y  = min(sy2, oy2)
                if enter_y < exit_y:
                    if ay <= by:
                        return 1, (sx, enter_y), (sx, exit_y)
                    else:
                        return 1, (sx, exit_y), (sx, enter_y)


        elif ay == by:
            sy = ay
            sx1, sx2 = min(ax, bx), max(ax, bx)
            if oy1 < sy < oy2 and sx1 < ox2 and sx2 > ox1:
                
                if in_obstacle(bx, by):
                    enter_pt = (ox1, sy) if ax <= bx else (ox2, sy)
                    return 2, enter_pt, (bx, by)

                if in_obstacle(ax, ay):
                    exit_pt = (ox2, sy) if ax <= bx else (ox1, sy)
                    return 3, (ax, ay), exit_pt

                enter_x = max(sx1, ox1)
                exit_x  = min(sx2, ox2)
                if enter_x < exit_x:
                    if ax <= bx:
                        return 1, (enter_x, sy), (exit_x, sy)
                    else:
                        return 1, (exit_x, sy), (enter_x, sy)

        return 0, None, None
 
    def _detour_around(self, kind, enter_pt, exit_pt,
                       ox1, oy1, ox2, oy2, coords):
        
        ex, ey = enter_pt
        fx, fy = exit_pt
 
        if kind == 'v':
            vX = ex
            left_extra  = 2 * (vX - ox1)
            right_extra = 2 * (ox2 - vX)
            side_x = ox1 if left_extra <= right_extra else ox2
            pts = [(side_x, ey), (side_x, fy), (fx, fy)]
 
        else:
            hY = ey
            top_extra    = 2 * (oy2 - hY)
            bottom_extra = 2 * (hY - oy1)
            side_y = oy2 if top_extra <= bottom_extra else oy1
            pts = [(ex, side_y), (fx, side_y), (fx, fy)]
 
        result = []
        for pt in pts:
            pid = self._get_or_create_coord(coords, pt)
            if not result or result[-1] != pid:
                result.append(pid)
        return result
 
    def _reroute_one_pair(self, v, h, coords, extra):
        vX, vY = coords[v]
        hX, hY = coords[h]
        n_o = 0

        is_vertical   = (vX == hX)
        is_horizontal = (vY == hY)
        
        if is_vertical:
            segments = [('v', vX, vY, hX, hY)]
        elif is_horizontal:
            segments = [('h', vX, vY, hX, hY)]
        else:
            segments = [
                ('v', vX, vY, vX, hY),
                ('h', vX, hY, hX, hY)
            ]

        point_seq = [v]

        for seg in segments:
            kind, ax, ay, bx, by = seg
            skip_end = False

            hits = []
            for ob_idx, ob in enumerate(self.obstacle):
                x1, y1, x2, y2 = ob
                flag, enter_pt, exit_pt = self._seg_cross_obstacle(
                    ax, ay, bx, by, x1, y1, x2, y2)
                if flag > 0:
                    #print(f"Segment {seg} hits obstacle {ob} at enter {enter_pt} and exit {exit_pt}")
                    dist = abs(enter_pt[0] - ax) + abs(enter_pt[1] - ay)
                    hits.append((dist, flag, ob_idx, enter_pt, exit_pt))

            hits.sort(key=lambda x: x[0])

            if not hits:
                end_id = self._get_or_create_coord(coords, (bx, by))
                if point_seq[-1] != end_id:
                    point_seq.append(end_id)
                continue
            
            for (_, flag, ob_idx, enter_pt, exit_pt) in hits:
                x1, y1, x2, y2 = self.obstacle[ob_idx]
                ox1, ox2 = min(x1, x2), max(x1, x2)
                oy1, oy2 = min(y1, y2), max(y1, y2)
                
                if flag == 1:
                    enter_id = self._get_or_create_coord(coords, enter_pt)
                    if point_seq[-1] != enter_id:
                        point_seq.append(enter_id)
                    detour_ids = self._detour_around(
                        kind, enter_pt, exit_pt,
                        ox1, oy1, ox2, oy2, coords
                    )
                    point_seq.extend(detour_ids)
                    skip_end = False
                    n_o += 1
                    
                elif flag == 2:
                    enter_id = self._get_or_create_coord(coords, enter_pt)
                    if point_seq[-1] != enter_id:
                        point_seq.append(enter_id) 
                                
                    last_pt = enter_pt   
                    skip_end = True         
                elif flag == 3:
                    corner_pt = (exit_pt[0],last_pt[1])
                    corner_id = self._get_or_create_coord(coords, corner_pt)
                    if point_seq[-1] != corner_id:
                        point_seq.append(corner_id)
                    exit_id = self._get_or_create_coord(coords, exit_pt)
                    if point_seq[-1] != exit_id:
                        point_seq.append(exit_id)
                    skip_end = False
                    
                    
            if not skip_end:
                end_id = self._get_or_create_coord(coords, (bx, by))
                if point_seq[-1] != end_id:
                    point_seq.append(end_id)
            
        clean = [point_seq[0]]
        for pid in point_seq[1:]:
            if pid != clean[-1]:
                clean.append(pid)

        result = []
        for i in range(len(clean) - 1):
            a, b_node = clean[i], clean[i + 1]
            if a != b_node:
                result.append((a, b_node) + tuple(extra))

        return result, n_o

    
    def reroute_around_obstacles(self, coords, connections):
        
        if self.ob_coord_base is None:
            coords, _ = self.insert_obstacle_coords(coords)
 
        new_connections = []
        n_overlap = 0
        for pair in connections:
            v, h = pair[0], pair[1]
            extra = pair[2:]
            rerouted, n_o = self._reroute_one_pair(v, h, coords, extra)
            new_connections.extend(rerouted)
            n_overlap += n_o
 
        new_connections = [p for p in new_connections if p[0] != p[1]]
        seen = set()
        deduped = []
        for p in new_connections:
            key = (p[0], p[1])
            if key not in seen:
                seen.add(key)
                deduped.append(p)
        return coords, deduped
 

    def two_lines_inter(self, v_line, h_line):
        v_x1, v_y1, v_x2, v_y2 = v_line
        h_x1, h_y1, h_x2, h_y2 = h_line
        if (h_y1 >= v_y1) and (h_y1 <= v_y2) and \
                (v_x1 >= h_x1) and (v_x1 <= h_x2):
            return True, (v_x1, h_y1)
        else:
            return False, (-1, -1)

    def check_horizontal_vertical_intersection_strict(self, horizontal_line, vertical_line):
        h_x1, h_y1, h_x2, h_y2 = horizontal_line
        v_x1, v_y1, v_x2, v_y2 = vertical_line
        h_x_min, h_x_max = min(h_x1, h_x2), max(h_x1, h_x2)
        v_y_min, v_y_max = min(v_y1, v_y2), max(v_y1, v_y2)
        if (h_x_min < v_x1 < h_x_max) and (v_y_min < h_y1 < v_y_max):
            return True, (v_x1, h_y1)
        else:
            return False, None

    def detect_intersections(self, v, h):
        intersect_ob = []
        vX, vY = v
        hX, hY = h
        wire_l = abs(vX - hX) + abs(vY - hY)
        v_line = (vX, vY, vX, hY)
        h_line = (hX, hY, vX, hY)
        for i in range(len(self.obstacle)):
            ob = self.obstacle[i]
            x1, y1, x2, y2 = ob
            h1 = (x1, y1, x2, y1)
            h2 = (x1, y2, x2, y2)
            v1 = (x1, y1, x1, y2)
            v2 = (x2, y1, x2, y2)
            if vY != hY and vX != hX:
                flag1, p1 = self.check_horizontal_vertical_intersection_strict(h1, v_line)
                flag2, p2 = self.check_horizontal_vertical_intersection_strict(h2, v_line)
                if flag1 and flag2:
                    p = min(abs(x1 - vX) / wire_l, abs(x2 - vX) / wire_l)
                    ob_l = min(abs(x1 - vX), abs(x2 - vX))
                    intersect_ob.append((i, p, ob_l))
                elif flag1 or flag2:
                    p = min(abs(x1 - vX) / wire_l, abs(x2 - vX) / wire_l,
                            abs(y1 - hY) / wire_l, abs(y2 - hY))
                    intersect_ob.append((i, p, 0.0))
                else:
                    flag3, p3 = self.check_horizontal_vertical_intersection_strict(h_line, v1)
                    flag4, p4 = self.check_horizontal_vertical_intersection_strict(h_line, v2)
                    if vX != x1 and vX != x2:
                        if flag3 and flag4:
                            p = min(abs(y1 - hY) / wire_l, abs(y2 - hY) / wire_l)
                            ob_l = min(abs(y1 - hY), abs(y2 - hY))
                            intersect_ob.append((i, p, ob_l))
                    else:
                        if flag3 or flag4:
                            p = min(abs(y1 - hY) / wire_l, abs(y2 - hY) / wire_l)
                            intersect_ob.append((i, p, 0.0))
            elif vY == hY:
                flag3, p3 = self.check_horizontal_vertical_intersection_strict(h_line, v1)
                flag4, p4 = self.check_horizontal_vertical_intersection_strict(h_line, v2)
                if flag3 and flag4:
                    p = min(abs(y1 - hY) / wire_l, abs(y2 - hY) / wire_l)
                    ob_l = min(abs(y1 - hY), abs(y2 - hY))
                    intersect_ob.append((i, p, ob_l))
            elif vX == hX:
                flag1, p1 = self.check_horizontal_vertical_intersection_strict(h1, v_line)
                flag2, p2 = self.check_horizontal_vertical_intersection_strict(h2, v_line)
                if flag1 and flag2:
                    p = min(abs(x1 - vX) / wire_l, abs(x2 - vX) / wire_l)
                    ob_l = min(abs(x1 - vX), abs(x2 - vX))
                    intersect_ob.append((i, p, ob_l))
        return intersect_ob

    def find_intersections(self, coords, connections):
        steiner_points = set()
        v_segs = []   # 竖直段: (x, y_min, y_max)
        h_segs = []   # 水平段: (y, x_min, x_max)
        for conn in connections:
            p, q = conn[0], conn[1]
            px, py = coords[p]
            qx, qy = coords[q]
            if abs(px - qx) < 1e-9:          # 竖直段
                v_segs.append((px, min(py, qy), max(py, qy)))
            elif abs(py - qy) < 1e-9:        # 水平段
                h_segs.append((py, min(px, qx), max(px, qx)))
        for (vx, vy1, vy2) in v_segs:
            for (hy, hx1, hx2) in h_segs:
                if hx1 < vx < hx2 and vy1 < hy < vy2:
                    steiner_points.add((vx, hy))
        return steiner_points
 
    def insert_steiner(self, steiner_points, coordinates, pairs):
        n = len(coordinates)
        for steiner in steiner_points:
            if steiner not in coordinates.values():
                coordinates[n] = steiner
                n += 1
        new_coords = {v: k for k, v in coordinates.items()}
 
        merged = defaultdict(list)
 
        for pair in pairs:
            v, h = pair[0], pair[1]
            extra = pair[2:]
            l_val = float(extra[1]) if len(extra) >= 2 else 0.0
            b_val = extra[0]   if len(extra) >= 1 else 0.0
 
            vX, vY = coordinates[v]
            hX, hY = coordinates[h]
 
            mid_pts = []
 
            if vX == hX:
                y_lo, y_hi = min(vY, hY), max(vY, hY)
                for pt in coordinates.values():
                    if abs(pt[0] - vX) < 1e-9 and y_lo < pt[1] < y_hi:
                        mid_pts.append(pt)
                mid_pts.sort(key=lambda p: p[1], reverse=(vY > hY))
 
            elif vY == hY:
                x_lo, x_hi = min(vX, hX), max(vX, hX)
                for pt in coordinates.values():
                    if abs(pt[1] - vY) < 1e-9 and x_lo < pt[0] < x_hi:
                        mid_pts.append(pt)
                mid_pts.sort(key=lambda p: p[0], reverse=(vX > hX))
 
            else:
                mid_pts = []
 
            seq = [v] + [new_coords[pt] for pt in mid_pts] + [h]
 
            for i in range(len(seq) - 1):
                a, b_node = seq[i], seq[i + 1]
                if a == b_node:
                    continue
                key = (a, b_node)
                if not merged[key]:
                    merged[key].append(l_val)
                    merged[key].append(b_val)
                else:
                    merged[key][0] = min(merged[key][0], l_val)
                    merged[key].append(b_val)
 
        result = []
        for (v, h), lb in merged.items():
            result.append(tuple([int(v), int(h)] + lb))
        return coordinates, result


    def find_buffer_positions(self, coords, connections, T):
        b_index = len(coords)
        connections_with_b = []
        for pair in connections:
            v, h = pair[0], pair[1]
            vX, vY = coords[v]
            hX, hY = coords[h]
            v_length = abs(vY - hY)
            h_length = abs(vX - hX)
            print(pair)
            
            l = pair[2]
            b = pair[3:]
            b_sin = b_cos = 0
            for i in b:
                b_sin += math.sin(i)
                b_cos += math.cos(i)
            if abs(b_cos) < 1e-12:
                b_cos = 1e-12
            
            start_position = (-math.atan(b_sin / b_cos) / 2 / math.pi + 0.25) * T
            
            if (v_length + h_length) < start_position:
                connections_with_b.append((v, h))
                continue
            buffer_num = int((v_length + h_length - float(start_position)) // T  + 1)
            for i in range(buffer_num):
                buffer_position = float(start_position) + T * i
                if buffer_position <= v_length:
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
    
    
    def merge_pairs_to_paths(self, pairs, coords, sink_num):
        from collections import defaultdict

        graph = defaultdict(list)
        edge_info = {}
        for p in pairs:
            v, h = p[0], p[1]
            graph[v].append(h)
            graph[h].append(v)
            edge_info[(v, h)] = p
            edge_info[(h, v)] = p

        def is_endpoint(node):
            return node < sink_num or len(graph[node]) != 2

        visited_edges = set()
        paths = []

        for start in list(graph.keys()):
            if not is_endpoint(start):
                continue
            for nxt in graph[start]:
                if (start, nxt) in visited_edges:
                    continue

                path = []
                prev, curr = start, nxt

                while True:
                    p = edge_info.get((prev, curr))
                    if p is None:
                        break
                    path.append((prev, curr) + tuple(p[2:]))
                    visited_edges.add((prev, curr))
                    visited_edges.add((curr, prev))

                    if is_endpoint(curr):
                        break

                    neighbors = graph[curr]
                    next_node = neighbors[0] if neighbors[1] == prev else neighbors[1]
                    prev, curr = curr, next_node

                if path:
                    paths.append(path)

        return paths
    
    def find_buffer_positions_on_paths(self, coords, paths, T, original_pairs=None):
        b_index = len(coords)
        new_connections = []

        for path in paths:
            if not path:
                continue

            point_seq = [path[0][0]]  # 起点
            for edge in path:
                point_seq.append(edge[1])

            seg_lens = []
            for i in range(len(point_seq) - 1):
                x1, y1 = coords[point_seq[i]]
                x2, y2 = coords[point_seq[i+1]]
                seg_lens.append(abs(x1-x2) + abs(y1-y2))
            total_len = sum(seg_lens)

            b = list(path[0][2:]) if len(path[0]) > 2 else []
            b_sin = b_cos = 0
            for val in b:
                b_sin += math.sin(val)
                b_cos += math.cos(val)
            if abs(b_cos) < 1e-12:
                b_cos = 1e-12
            start_position = ( -math.atan(b_sin / b_cos) / 2 / math.pi + 0.25) * T
            
            start_position = start_position % T

            buffer_positions = []
            if total_len >= start_position:
                buffer_num = int((total_len - start_position) // T + 1)
                for i in range(buffer_num):
                    buffer_positions.append(start_position + i * T)

            
            all_nodes = [point_seq[0]]
            cumulative = 0.0
            buf_idx = 0

            for seg_i, (pa, pb) in enumerate(zip(point_seq[:-1], point_seq[1:])):
                x1, y1 = coords[pa]
                x2, y2 = coords[pb]
                seg_len = seg_lens[seg_i]
                seg_start = cumulative

                while buf_idx < len(buffer_positions):
                    pos = buffer_positions[buf_idx]
                    if pos > seg_start + seg_len + 1e-9:
                        break
                    d = pos - seg_start
                    d = max(0, min(d, seg_len))
                    if seg_len > 1e-9:
                        t = d / seg_len
                    else:
                        t = 0
                    bx = x1 + t * (x2 - x1)
                    by = y1 + t * (y2 - y1)
                    coords[b_index] = (bx, by)
                    all_nodes.append(b_index)
                    b_index += 1
                    buf_idx += 1

                all_nodes.append(pb)
                cumulative += seg_len

            for i in range(len(all_nodes) - 1):
                a, b_node = all_nodes[i], all_nodes[i+1]
                if a != b_node:
                    new_connections.append((a, b_node))

        return coords, new_connections


    def point_in_obstacle(self, pt):
        x, y = pt
        for (x1, y1, x2, y2) in self.obstacle:
            if min(x1, x2) < x < max(x1, x2) and \
            min(y1, y2) < y < max(y1, y2):
                return True
        return False
            
    

    def build_tree(self, pair_with_steiner, coords_with_steiner, sink_cap, wire_capacitance, wire_resistance, N):
        visited = [False] * len(coords_with_steiner)
        queue = deque([0])
        visited[0] = True
        root = Node(name=0)
        nodes = {0: root}
        while queue:
            current = queue.popleft()
            current_tree_node = nodes[current]
            for pair in pair_with_steiner:
                v, h = pair
                #print(v,h)
                vX, vY = coords_with_steiner[v]
                hX, hY = coords_with_steiner[h]
                v_length = abs(vY - hY)
                h_length = abs(vX - hX)
                sink_capacitance = 0
                buffer_position = False
                if v == current and visited[h] == False:
                    visited[h] = True
                    queue.append(h)
                    wire_length = v_length + h_length
                    point_capacitance = wire_capacitance * wire_length
                    point_resistance = wire_resistance * wire_length
                    if h < len(sink_cap):
                        sink_capacitance = sink_cap[h]
                    elif h >= N:
                        buffer_position = True
                    child_node = Node(name=h, resistance=point_resistance, capacitance=point_capacitance,
                                      sink_capacitance=sink_capacitance, buffer_position=buffer_position)
                    child_node.wl = wire_length
                    current_tree_node.children.append(child_node)
                    nodes[h] = child_node
                elif h == current and visited[v] == False:
                    visited[v] = True
                    queue.append(v)
                    wire_length = v_length + h_length
                    point_capacitance = wire_capacitance * wire_length
                    point_resistance = wire_resistance * wire_length
                    if v < len(sink_cap):
                        sink_capacitance = sink_cap[v]
                    elif v >= N:
                        buffer_position = True
                    child_node = Node(name=v, resistance=point_resistance, capacitance=point_capacitance,
                                      sink_capacitance=sink_capacitance, buffer_position=buffer_position)
                    child_node.wl = wire_length
                    current_tree_node.children.append(child_node)
                    nodes[v] = child_node
        return root, nodes


class Evaluator:
    def __init__(self, degree):
        self.wire_resistance = 0.003000
        self.wire_capacitance = 2.000000e-15
        if degree <= 10:
            self.buffer_delay = 3e-21
            self.buffer_capacitance = 5e-16
            self.buffer_resistance = 0.0002
        elif degree <= 20:
            self.buffer_delay = 3e-21
            self.buffer_capacitance = 4e-16
            self.buffer_resistance = 0.00015
        else:
            self.buffer_delay = 3e-21
            self.buffer_capacitance = 3e-16
            self.buffer_resistance = 0.00012
        #self.obstacle = obstacle
        self.tree = ObstacleTree()

    def eval_batch(self, input_batch, input_obstacle_batch, output_batch, degree, obstacle_batch):
        lengths = []
        slacks = []
        skews = []
        n_os = []
        batch_size = input_batch.shape[0]
        for i in range(batch_size):
            coords = {j: tuple(input_batch[i, j, :2].tolist()) for j in range(degree)}
            sink_cap = {j: input_batch[i, j, 2].item() / 1e15 for j in range(degree)}
            connections = [(output_batch[i, 3*k].item(), output_batch[i, 3*k+1].item(),
                            output_batch[i, 3*k+2].item()) for k in range(degree - 1)]
            obs = input_obstacle_batch[i]
            valid = obs[obs[:, 0] >= 0]
            obstacles = [tuple(row) for row in valid]
            self.tree.input_ob(obstacles)
            obstacle_sin = [obstacle_batch[i,j,:] for j in range(degree - 1)]
            slack, skew, length = self.eval_func(coords, sink_cap, connections,obstacle_sin)
            lengths.append(length)
            slacks.append(slack)
            skews.append(skew)
            #n_os.append(n_o)
        return np.array(slacks) * 1e18, np.array(skews) * 1e18, np.array(lengths)
        
    def eval_func(self, coords, sink_cap, connections, obsin):
       
        T = math.sqrt(2 * self.buffer_capacitance * self.buffer_resistance /
                      self.wire_resistance / self.wire_capacitance)
        terminal_set = {coords[i] for i in coords}
        coords_with_sin, connections_with_sin = self.tree.change_sin(coords, connections, obsin, T)
        coords_with_o, connections_with_o = self.tree.reroute_around_obstacles(coords_with_sin, connections_with_sin)
        
        steiner_points = self.tree.find_intersections(coords_with_o, connections_with_o)
        
        coords_with_s, pairs_with_s = self.tree.insert_steiner(
            steiner_points, coords_with_o, connections_with_o)
        num_with_steiner = len(coords_with_s)
        sink_num = len(sink_cap)
        
        merged_paths = self.tree.merge_pairs_to_paths(pairs_with_s, coords_with_s, sink_num)
        coords_with_b, pairs_with_b = self.tree.find_buffer_positions_on_paths(coords_with_s, merged_paths, T, pairs_with_s)

        num_buffer = len(coords_with_b) - len(sink_cap)

        root, nodes = self.tree.build_tree(
            pairs_with_b, coords_with_b, sink_cap,
            self.wire_capacitance, self.wire_resistance, num_with_steiner)
        
        buffer = BufferInsertion(root, self.buffer_delay,
                                 self.buffer_capacitance, self.buffer_resistance)
        depth = buffer.calculate_depth(root)
        sink_num = len(sink_cap)
        buffer.optimal_buffer_insertion(nodes, depth, sink_num, num_with_steiner)

        sink_delay = [0] * (sink_num - 1)
        wirelength = 0
        for i in nodes:
            nodes[i].opt_delay = buffer.compute_elmore_delay(nodes[i])
            wirelength += nodes[i].wl
        for i in range(1, sink_num):
            sink = nodes[i]
            path = buffer.find_path(root, sink)
            for inner in range(1, len(path)):
                sink_delay[i - 1] += nodes[path[inner]].opt_delay
        delayMax = max(sink_delay)
        delayMin = min(sink_delay)
       
        return delayMax, delayMax - delayMin, wirelength
    
