import math
import random
import heapq
import copy
from collections import deque
 

REQUEST_TEMPLATE = {
    "request_id"       : "",
    "name"             : "",
    "role"             : "",         
    "request_type"     : "",         
                                      
    "category"         : "",         
                                     
    "current_location" : "",
    "destination"      : "",
    "preferred_slot"   : None,        # 1-4
    "severity"         : 0,           # 1-10
    "time_sensitivity" : 0,           # 1-10
    "crowd_level"      : 0,           # 1-10
    "group_id"         : "",
    "query"            : "",
    "eligibility_claim": False,
    "description_note" : ""
}
 
ROUTER_OUTPUT_TEMPLATE = {
    "request_id"       : "",
    "selected_pipeline": [],
    "needs_ann"        : False,
    "needs_logic"      : False,
    "needs_csp"        : False,
    "needs_search"     : False
}
 
PRIORITY_OUTPUT_TEMPLATE = {
    "binary_priority"  : "",    # urgent | not_urgent
    "final_priority"   : "",    # low | normal | high | urgent
    "confidence"       : 0.0
}
 
LOGIC_OUTPUT_TEMPLATE = {
    "allowed"     : False,
    "entailed"    : False,
    "explanation" : ""
}
 
CSP_OUTPUT_TEMPLATE = {
    "decision"      : "",
    "assigned_room" : "",
    "assigned_slot" : None,
    "destination"   : "",
    "notes"         : ""
}
 
SEARCH_OUTPUT_TEMPLATE = {
    "algorithm_used" : "",
    "path"           : [],
    "cost"           : 0,
    "steps"          : 0
}
 
FINAL_RESPONSE_TEMPLATE = {
    "request_id"  : "",
    "decision"    : "",
    "priority"    : {},
    "eligibility" : {},
    "assignment"  : {},
    "route"       : {},
    "message"     : ""
}
 

FEATURE_ORDER = [
    "Role", "RequestType", "Severity",
    "TimeSensitivity", "CrowdLevel", "Distance", "Eligibility"
]
 
ROLE_ENCODING = {
    "student"   : 0,
    "instructor": 1,
    "staff"     : 2
}
 
REQUEST_TYPE_ENCODING = {
    "AI_Lab_Support"  : 0,
    "Viva_Scheduling" : 1,
    "Access_Request"  : 2,
    "Maintenance"     : 3,
    "Emergency_Help"  : 4
}
 
def encode_bool(val):
    return 1 if val else 0

VALID_ROLES = {"student", "instructor", "staff"}
 
VALID_REQUEST_TYPES = {
    "Navigation_Only",
    "Eligibility_Check",
    "Booking_or_Scheduling",
    "Urgent_Service_Request",
    "Full_Service_Request"
}
 
VALID_CATEGORIES = {
    "AI_Lab_Support",
    "Viva_Scheduling",
    "Access_Request",
    "Maintenance",
    "Emergency_Help"
}
 
VALID_LOCATIONS = {
    "Main_Gate", "Parking", "Admin_Block", "Student_Services",
    "Exam_Hall", "Seminar_Room", "Library", "AI_Lab",
    "Science_Block", "Cafeteria", "Hostel", "Medical_Center",
    "Bus_Stop"
}
 
VALID_SLOTS = {1, 2, 3, 4}
 
NORMALIZE_MAP = {
    "ai lab"           : "AI_Lab",
    "ai_lab"           : "AI_Lab",
    "hostel"           : "Hostel",
    "main gate"        : "Main_Gate",
    "main_gate"        : "Main_Gate",
    "science block"    : "Science_Block",
    "library"          : "Library",
    "cafeteria"        : "Cafeteria",
    "parking"          : "Parking",
    "exam hall"        : "Exam_Hall",
    "exam_hall"        : "Exam_Hall",
    "admin block"      : "Admin_Block",
    "admin_block"      : "Admin_Block",
    "seminar room"     : "Seminar_Room",
    "medical center"   : "Medical_Center",
    "bus stop"         : "Bus_Stop",
    "student services" : "Student_Services",
    "urgent_service_request" : "Urgent_Service_Request",
    "full_service_request"   : "Full_Service_Request",
    "navigation_only"        : "Navigation_Only",
    "eligibility_check"      : "Eligibility_Check",
    "booking_or_scheduling"  : "Booking_or_Scheduling",
}
 
def normalize_value(val):
    if not isinstance(val, str):
        return val
    lower = val.strip().lower()
    return NORMALIZE_MAP.get(lower, val.strip())
 
 
def validate_request(raw):
  
    errors = []
 
    if not raw.get("name"):
        errors.append("Missing field: name")
    if not raw.get("role"):
        errors.append("Missing field: role")
    elif raw["role"] not in VALID_ROLES:
        errors.append(f"Invalid role '{raw['role']}'. Must be one of {VALID_ROLES}")
    if not raw.get("request_type"):
        errors.append("Missing field: request_type")
    elif raw["request_type"] not in VALID_REQUEST_TYPES:
        errors.append(f"Invalid request_type '{raw['request_type']}'")
 
    rt = raw.get("request_type", "")
 
    if rt == "Navigation_Only":
        if not raw.get("current_location"):
            errors.append("Navigation_Only requires: current_location")
        if not raw.get("destination"):
            errors.append("Navigation_Only requires: destination")
 
    elif rt == "Eligibility_Check":
        if not raw.get("query"):
            errors.append("Eligibility_Check requires: query")
 
    elif rt in ("Booking_or_Scheduling", "Urgent_Service_Request", "Full_Service_Request"):
        if not raw.get("category"):
            errors.append(f"{rt} requires: category")
        elif raw["category"] not in VALID_CATEGORIES:
            errors.append(f"Invalid category '{raw['category']}'")
        if not raw.get("current_location"):
            errors.append(f"{rt} requires: current_location")
 
    if raw.get("preferred_slot") is not None:
        try:
            s = int(raw["preferred_slot"])
            if s not in VALID_SLOTS:
                errors.append(f"preferred_slot must be in {VALID_SLOTS}, got {s}")
        except (ValueError, TypeError):
            errors.append("preferred_slot must be a number")
 
    for field in ("severity", "time_sensitivity", "crowd_level"):
        val = raw.get(field, 0)
        if val:
            try:
                v = int(val)
                if not (1 <= v <= 10):
                    errors.append(f"{field} must be between 1 and 10, got {v}")
            except (ValueError, TypeError):
                errors.append(f"{field} must be numeric")
 
    for loc_field in ("current_location", "destination"):
        v = raw.get(loc_field)
        if v and v not in VALID_LOCATIONS:
            errors.append(f"Unknown location '{v}' in {loc_field}")
 
    return (len(errors) == 0), errors
 
 
_req_counter = [100]
 
def preprocess_request(raw):
    
    normalized = {}
    for k, v in raw.items():
        if isinstance(v, str):
            normalized[k] = normalize_value(v)
        else:
            normalized[k] = v
 
    is_valid, errors = validate_request(normalized)
    if not is_valid:
        return None, None, errors
 
    _req_counter[0] += 1
    req_id = f"REQ{_req_counter[0]}"
 
    request_obj = dict(REQUEST_TEMPLATE)
    request_obj.update({
        "request_id"       : req_id,
        "name"             : normalized.get("name", ""),
        "role"             : normalized.get("role", ""),
        "request_type"     : normalized.get("request_type", ""),
        "category"         : normalized.get("category", ""),
        "current_location" : normalized.get("current_location", ""),
        "destination"      : normalized.get("destination", ""),
        "preferred_slot"   : int(normalized["preferred_slot"]) if normalized.get("preferred_slot") else None,
        "severity"         : int(normalized.get("severity", 0) or 0),
        "time_sensitivity" : int(normalized.get("time_sensitivity", 0) or 0),
        "crowd_level"      : int(normalized.get("crowd_level", 0) or 0),
        "group_id"         : normalized.get("group_id", ""),
        "query"            : normalized.get("query", ""),
        "eligibility_claim": bool(normalized.get("eligibility_claim", False)),
        "description_note" : normalized.get("description_note", "")
    })
 
    rt = request_obj["request_type"]
    pipeline_flags = {
        "needs_ann"   : rt in ("Urgent_Service_Request", "Full_Service_Request"),
        "needs_logic" : rt in ("Eligibility_Check", "Booking_or_Scheduling",
                               "Urgent_Service_Request", "Full_Service_Request"),
        "needs_csp"   : rt in ("Booking_or_Scheduling", "Urgent_Service_Request",
                               "Full_Service_Request"),
        "needs_search": rt in ("Navigation_Only", "Full_Service_Request")
    }
 
    return request_obj, pipeline_flags, []
 
 
 
print("\n Test: Preprocessing Module ")
raw_input = {
    "name"             : "Ali",
    "role"             : "student",
    "request_type"     : "Full_Service_Request",
    "category"         : "AI_Lab_Support",
    "current_location" : "Hostel",
    "preferred_slot"   : 2,
    "severity"         : 8,
    "time_sensitivity" : 9,
    "crowd_level"      : 5,
    "description_note" : "Need urgent help before practical evaluation."
}
 
req_obj, flags, errors = preprocess_request(raw_input)
 
if errors:
    print("VALIDATION ERRORS:", errors)
else:
    print("Request Object:")
    for k, v in req_obj.items():
        print(f"  {k:20s}: {v}")
    print("\nPipeline Flags:")
    for k, v in flags.items():
        print(f"  {k:15s}: {v}")

def route_request(request_obj, pipeline_flags):
 
    rt = request_obj["request_type"]
 
    pipeline_map = {
        "Navigation_Only"        : ["Search"],
        "Eligibility_Check"      : ["Logic_KB"],
        "Booking_or_Scheduling"  : ["Logic_KB", "CSP"],
        "Urgent_Service_Request" : ["ANN", "Logic_KB", "CSP"],
        "Full_Service_Request"   : ["ANN", "Logic_KB", "CSP", "Search"]
    }
 
    if rt not in pipeline_map:
        return None, f"Unknown request_type: {rt}"
 
    pipeline = pipeline_map[rt]
 
    router_output = dict(ROUTER_OUTPUT_TEMPLATE)
    router_output.update({
        "request_id"       : request_obj["request_id"],
        "selected_pipeline": pipeline,
        "needs_ann"        : pipeline_flags["needs_ann"],
        "needs_logic"      : pipeline_flags["needs_logic"],
        "needs_csp"        : pipeline_flags["needs_csp"],
        "needs_search"     : pipeline_flags["needs_search"]
    })
 
    return router_output, None
 
 
 
print("\n Test: Request Router ")
router_out, err = route_request(req_obj, flags)
if err:
    print("Router Error:", err)
else:
    print("Router Output:")
    for k, v in router_out.items():
        print(f"  {k:20s}: {v}")

NODE_COORDS = {
    "Main_Gate"        : (0, 4),
    "Bus_Stop"         : (0, 1),
    "Medical_Center"   : (1, 1),
    "Parking"          : (2, 4),
    "Hostel"           : (2, 0),
    "Admin_Block"      : (3, 5),
    "Cafeteria"        : (4, 1),
    "Student_Services" : (6, 5),
    "Library"          : (6, 2),
    "Science_Block"    : (7, 1),
    "Exam_Hall"        : (8, 5),
    "AI_Lab"           : (9, 2),
    "Seminar_Room"     : (10, 4),
}
 


CAMPUS_GRAPH_WEIGHTED = {
    "Main_Gate"        : [("Bus_Stop", 1),    ("Parking", 2),    ("Admin_Block", 4),   ("Hostel", 5)],
    "Bus_Stop"         : [("Main_Gate", 1),   ("Hostel", 5), ("Parking", 2)],
    "Parking"          : [("Main_Gate", 2),   ("Science_Block", 3),("Bus_Stop", 2) ],
    "Admin_Block"      : [("Main_Gate", 4),   ("Student_Services", 1),("Exam_Hall", 2)],
    "Student_Services" : [("Admin_Block", 1), ("Library", 2)],
    "Exam_Hall"        : [("Seminar_Room", 1),("Science_Block", 3), ("Admin_Block", 2)],
    "Seminar_Room"     : [("Exam_Hall", 1),  ("Science_Block", 2)],
    "Library"          : [("Student_Services", 2), ("Cafeteria", 2), ("AI_Lab", 3)],
    "AI_Lab"           : [("Library", 3),  ("Science_Block", 1)],
    "Science_Block"    : [("AI_Lab", 1),     ("Seminar_Room", 2),    ("Cafeteria", 3),    ("Exam_Hall", 3),   ("Parking", 3)],
    "Cafeteria"        : [("Library", 2),    ("Science_Block", 3), ("Hostel", 2)],
    "Medical_Center"   : [("Bus_Stop", 2) ,  ("Hostel", 3)],
    "Hostel"           : [("Medical_Center", 3), ("Cafeteria", 2), ("Main_Gate", 5)],
}
 
CAMPUS_GRAPH_UNWEIGHTED = {
    "Bus_Stop"         : ["Main_Gate",    "Parking",          "Medical_Center"],
    "Main_Gate"        : ["Bus_Stop",     "Admin_Block",     "Parking",    "Hostel"],
    "Admin_Block"      : ["Main_Gate",    "Student_Services","Exam_Hall"],
    "Student_Services" : ["Admin_Block", "Library"],
    "Exam_Hall"        : ["Admin_Block",  "Seminar_Room","Science_Block"],
    "Seminar_Room"     : ["Student_Services","Exam_Hall"],
    "Parking"          : ["Main_Gate",    "Science_Block",   "Bus_Stop"],
    "Hostel"           : ["Main_Gate",  "Cafeteria", "Medical_Center"],
    "Science_Block"    : ["Parking",         "Exam_Hall",  "Seminar_Room",
                          "AI_Lab",          "Cafeteria"],
    "Cafeteria"        : ["Hostel",       "Science_Block",   "Library"],
    "AI_Lab"           : ["Science_Block", "Library"],
    "Library"          : ["AI_Lab"],
    "Medical_Center"   : ["Bus_Stop","Hostel"],
}
 
# Admissible Heuristic 
HEURISTIC_TO_AILAB = {
    "AI_Lab"           : 0,
    "Science_Block"    : 1,
    "Seminar_Room"     : 2,
    "Library"          : 3,
    "Exam_Hall"        : 3,
    "Cafeteria"        : 4,
    "Hostel"           : 3,
    "Medical_Center"   : 4,
    "Student_Services" : 5,
    "Admin_Block"      : 6,
    "Parking"          : 7,
    "Bus_Stop"         : 5,
    "Main_Gate"        : 9,
}
 
 
def get_heuristic(node, goal, heuristic_map=None):
    if heuristic_map:
        return heuristic_map.get(node, 5)
    return 0
 
 
# ALGORITHM 1  BFS (Breadth-First Search)

def bfs(graph, start, goal):
    if start == goal:
        return [start], 0, 0
 
    queue    = deque([(start, [start])])
    visited  = {start}
    expanded = 0
 
    while queue:
        node, path = queue.popleft()
        expanded  += 1
        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                new_path = path + [neighbor]
                if neighbor == goal:
                    return new_path, len(new_path) - 1, expanded
                visited.add(neighbor)
                queue.append((neighbor, new_path))
 
    return None, float("inf"), expanded
 
 
# ALGORITHM 2 DFS (Depth-First Search)
def dfs(graph, start, goal):
    stack    = [(start, [start])]
    visited  = set()
    expanded = 0
 
    while stack:
        node, path = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        expanded += 1
 
        if node == goal:
            return path, len(path) - 1, expanded
 
        for neighbor in reversed(graph.get(node, [])):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))
 
    return None, float("inf"), expanded
 
 

# ALGORITHM 3  DLS (Depth-Limited Search)
def dls(graph, start, goal, limit=5):
    def _dls_rec(node, path, depth):
        if node == goal:
            return path
        if depth == 0:
            return None
        for neighbor in graph.get(node, []):
            if neighbor not in path:
                result = _dls_rec(neighbor, path + [neighbor], depth - 1)
                if result:
                    return result
        return None
 
    path = _dls_rec(start, [start], limit)
    if path:
        return path, len(path) - 1, limit
    return None, float("inf"), -1
 
 
# ALGORITHM 4  IDS (Iterative Deepening Search)
def ids(graph, start, goal, max_depth=20):
    for depth in range(max_depth + 1):
        path, cost, _ = dls(graph, start, goal, limit=depth)
        if path:
            return path, cost, depth
    return None, float("inf"), -1
 
 
# ALGORITHM 5  UCS (Uniform Cost Search)

def ucs(graph, start, goal):
    heap     = [(0, start, [start])]
    visited  = {}
    expanded = 0
 
    while heap:
        cost, node, path = heapq.heappop(heap)
        if node in visited and visited[node] <= cost:
            continue
        visited[node] = cost
        expanded     += 1
 
        if node == goal:
            return path, cost, expanded
 
        for neighbor, edge_cost in graph.get(node, []):
            new_cost = cost + edge_cost
            if neighbor not in visited or visited.get(neighbor, float("inf")) > new_cost:
                heapq.heappush(heap, (new_cost, neighbor, path + [neighbor]))
 
    return None, float("inf"), expanded
 
 
# ALGORITHM 6  Bidirectional BFS

def bidirectional_bfs(graph, start, goal):
    if start == goal:
        return [start], 0, 0
 
    fwd_visited = {start: [start]}
    bwd_visited = {goal:  [goal]}
    fwd_queue   = deque([start])
    bwd_queue   = deque([goal])
    expanded    = 0
 
    while fwd_queue or bwd_queue:
        # Expand forward frontier
        if fwd_queue:
            node = fwd_queue.popleft()
            expanded += 1
            for nb in graph.get(node, []):
                if nb not in fwd_visited:
                    fwd_visited[nb] = fwd_visited[node] + [nb]
                    fwd_queue.append(nb)
                if nb in bwd_visited:
                    path = fwd_visited[nb] + list(reversed(bwd_visited[nb][:-1]))
                    return path, len(path) - 1, expanded
 
        # Expand backward frontier
        if bwd_queue:
            node = bwd_queue.popleft()
            expanded += 1
            for nb in graph.get(node, []):
                if nb not in bwd_visited:
                    bwd_visited[nb] = bwd_visited[node] + [nb]
                    bwd_queue.append(nb)
                if nb in fwd_visited:
                    path = fwd_visited[nb] + list(reversed(bwd_visited[nb][:-1]))
                    return path, len(path) - 1, expanded
 
    return None, float("inf"), expanded
 
 

# ALGORITHM 7  Greedy Best-First Search

def greedy_bfs(graph, start, goal, heuristic_map=None):
    heap     = [(get_heuristic(start, goal, heuristic_map), start, [start])]
    visited  = set()
    expanded = 0
 
    while heap:
        _, node, path = heapq.heappop(heap)
        if node in visited:
            continue
        visited.add(node)
        expanded += 1
 
        if node == goal:
            # Compute actual path cost
            actual_cost = 0
            for i in range(len(path) - 1):
                for nb, c in graph.get(path[i], []):
                    if nb == path[i + 1]:
                        actual_cost += c
                        break
            return path, actual_cost, expanded
 
        for neighbor, _ in graph.get(node, []):
            if neighbor not in visited:
                h = get_heuristic(neighbor, goal, heuristic_map)
                heapq.heappush(heap, (h, neighbor, path + [neighbor]))
 
    return None, float("inf"), expanded
 
 
# ALGORITHM 8  A* (A-Star)
# f(n) = g(n) + h(n)

def astar(graph, start, goal, heuristic_map=None):

    h0   = get_heuristic(start, goal, heuristic_map)
    heap = [(h0, 0, start, [start])]  # (f, g, node, path)
    g_costs  = {start: 0}
    expanded = 0
 
    while heap:
        f, g, node, path = heapq.heappop(heap)
        if g > g_costs.get(node, float("inf")):
            continue
        expanded += 1
 
        if node == goal:
            return path, g, expanded
 
        for neighbor, cost in graph.get(node, []):
            new_g = g + cost
            if new_g < g_costs.get(neighbor, float("inf")):
                g_costs[neighbor] = new_g
                h = get_heuristic(neighbor, goal, heuristic_map)
                heapq.heappush(heap, (new_g + h, new_g, neighbor, path + [neighbor]))
 
    return None, float("inf"), expanded
 
 
# ALGORITHM 9 RBFS (Recursive Best-First Search)

def rbfs(graph, start, goal, heuristic_map=None):
 
    def _rbfs_inner(node, path, g, f_limit):
        h = get_heuristic(node, goal, heuristic_map)
        f = g + h
        if f > f_limit:
            return None, f
        if node == goal:
            return path, f
 
        successors = []
        for neighbor, cost in graph.get(node, []):
            if neighbor not in path:
                ng = g + cost
                nh = get_heuristic(neighbor, goal, heuristic_map)
                nf = ng + nh
                successors.append((nf, ng, neighbor, path + [neighbor]))
 
        if not successors:
            return None, float("inf")
 
        successors.sort(key=lambda x: x[0])
 
        while True:
            best_f, best_g, best_node, best_path = successors[0]
            if best_f > f_limit:
                return None, best_f
            alt_f = successors[1][0] if len(successors) > 1 else float("inf")
            result, new_f = _rbfs_inner(best_node, best_path, best_g,
                                        min(f_limit, alt_f))
            successors[0] = (new_f, best_g, best_node, best_path)
            successors.sort(key=lambda x: x[0])
            if result:
                return result, new_f
 
    result, _ = _rbfs_inner(start, [start], 0, float("inf"))
    if result:
        cost = 0
        for i in range(len(result) - 1):
            for nb, c in graph.get(result[i], []):
                if nb == result[i + 1]:
                    cost += c
                    break
        return result, cost, -1
    return None, float("inf"), -1
 

print("\n Campus Weighted Graph ")
for node, edges in CAMPUS_GRAPH_WEIGHTED.items():
    print(f"  {node:20s}: {edges}")
 
print("\n Admissible Heuristics to AI_Lab ")
for node, h in sorted(HEURISTIC_TO_AILAB.items(), key=lambda x: x[1]):
    print(f"  {node:20s}: h={h}")
 
 
def run_search_module(source, destination, graph_type="weighted"):
 
    result = dict(SEARCH_OUTPUT_TEMPLATE)
 
    if not source or not destination:
        result.update({"algorithm_used": "None", "path": [], "cost": -1, "steps": -1})
        return result
 
    if source == destination:
        result.update({"algorithm_used": "None", "path": [source], "cost": 0, "steps": 0})
        return result
 
    if graph_type == "unweighted":
        path, cost, expanded = bfs(CAMPUS_GRAPH_UNWEIGHTED, source, destination)
        algo = "BFS"
    else:
        path, cost, expanded = astar(
            CAMPUS_GRAPH_WEIGHTED, source, destination, HEURISTIC_TO_AILAB
        )
        algo = "A*"
        if path is None:
            path, cost, expanded = ucs(CAMPUS_GRAPH_WEIGHTED, source, destination)
            algo = "UCS (fallback)"
 
    if path is None:
        result.update({"algorithm_used": algo, "path": [], "cost": -1, "steps": -1})
        return result
 
    result.update({
        "algorithm_used": algo,
        "path"          : path,
        "cost"          : cost,
        "steps"         : len(path) - 1
    })
    return result
 
 
 
# Test: Search Module 
print("\n Test: Module 1 — Search Module ")
search_test_out = run_search_module("Hostel", "AI_Lab", graph_type="weighted")
print(f"Route: Hostel → AI_Lab")
print(f"  Algorithm : {search_test_out['algorithm_used']}")
print(f"  Path      : {' → '.join(search_test_out['path'])}")
print(f"  Cost      : {search_test_out['cost']}")
print(f"  Steps     : {search_test_out['steps']}")
 
print("\n Algorithm Comparison: Hostel → AI_Lab ")
src = "Hostel"
dst = "AI_Lab"
h   = HEURISTIC_TO_AILAB
 
print("\nUnweighted Graph (hop count as cost):")
print(f"  {'Algorithm':<22} {'Cost':>5}  {'Expanded':>9}  Path")
print("  " + " " * 75)
uw_tests = [
    ("BFS",               bfs(CAMPUS_GRAPH_UNWEIGHTED, src, dst)),
    ("DFS",               dfs(CAMPUS_GRAPH_UNWEIGHTED, src, dst)),
    ("DLS (limit=8)",     dls(CAMPUS_GRAPH_UNWEIGHTED, src, dst, limit=8)),
    ("IDS",               ids(CAMPUS_GRAPH_UNWEIGHTED, src, dst)),
    ("Bidirectional BFS", bidirectional_bfs(CAMPUS_GRAPH_UNWEIGHTED, src, dst)),
]
for name, (path, cost, exp) in uw_tests:
    p = " → ".join(path) if path else "No path"
    print(f"  {name:<22} {cost:>5}  {str(exp):>9}  {p}")
 
print("\nWeighted Graph (edge weights as cost):")
print(f"  {'Algorithm':<22} {'Cost':>5}  {'Expanded':>9}  Path")
print("  " + " " * 75)
wt_tests = [
    ("UCS",         ucs(CAMPUS_GRAPH_WEIGHTED, src, dst)),
    ("Greedy BFS",  greedy_bfs(CAMPUS_GRAPH_WEIGHTED, src, dst, h)),
    ("A*",          astar(CAMPUS_GRAPH_WEIGHTED, src, dst, h)),
    ("RBFS",        rbfs(CAMPUS_GRAPH_WEIGHTED, src, dst, h)),
]
for name, (path, cost, exp) in wt_tests:
    p = " → ".join(path) if path else "No path"
    print(f"  {name:<22} {cost:>5}  {str(exp):>9}  {p}")


# Knowledge Base 
KNOWLEDGE_BASE = {
    "facts": [
        # Students
        ("Student",    "Ali"),
        ("Student",    "Sara"),
        ("Student",    "Hamza"),
        # Prerequisites completed
        ("Completed",  "Ali",   "ProgrammingFundamentals"),
        ("Completed",  "Hamza", "ProgrammingFundamentals"),
        # Course enrollment
        ("Enrolled",   "Ali",   "AI"),
        ("Enrolled",   "Hamza", "AI"),
        ("Enrolled",   "Sara",  "SE"),
        # Instructors
        ("Teaches",    "DrKhan", "AI"),
        ("Teaches",    "DrAli",  "SE"),
        # Roles
        ("Role",       "Ali",    "student"),
        ("Role",       "Sara",   "student"),
        ("Role",       "Hamza",  "student"),
        ("Role",       "DrKhan", "instructor"),
        ("Role",       "DrAli",  "instructor"),
        ("Role",       "Ahmed",  "staff"),
        # Labs and Rooms
        ("LabExists",  "AI_Lab"),
        ("LabExists",  "SE_Lab"),
        ("RoomExists", "Seminar_Room"),
        ("RoomExists", "Exam_Hall"),
        # Slot availability
        ("SlotAvailable", "AI_Lab",       1),
        ("SlotAvailable", "AI_Lab",       2),
        ("SlotAvailable", "AI_Lab",       3),
        ("SlotAvailable", "AI_Lab",       4),
        ("SlotAvailable", "Exam_Hall",    1),
        ("SlotAvailable", "Exam_Hall",    2),
        ("SlotAvailable", "Exam_Hall",    3),
        ("SlotAvailable", "Seminar_Room", 1),
        ("SlotAvailable", "Seminar_Room", 2),
        # Staff authorization
        ("MaintenanceAuth", "Ahmed"),
    ],
    "rules": [
        # R1: Student(x) & Completed(x, ProgrammingFundamentals) => Eligible(x, AI)
        {
            "name"      : "R1_StudentCompletedEligible",
            "condition" : ("Completed", "?x", "ProgrammingFundamentals"),
            "conclusion": ("Eligible", "?x", "AI")
        },
        # R2: Teaches(x, AI) => Instructor(x, AI)
        {
            "name"      : "R2_TeachesImpliesInstructor",
            "condition" : ("Teaches", "?x", "AI"),
            "conclusion": ("Instructor", "?x", "AI")
        },
        # R3: Enrolled(x, AI) => UsesLab(x, AI_Lab)
        {
            "name"      : "R3_EnrolledUsesLab",
            "condition" : ("Enrolled", "?x", "AI"),
            "conclusion": ("UsesLab", "?x", "AI_Lab")
        },
        # R4: Instructor(x, AI) => UsesLab(x, AI_Lab)
        {
            "name"      : "R4_InstructorUsesLab",
            "condition" : ("Instructor", "?x", "AI"),
            "conclusion": ("UsesLab", "?x", "AI_Lab")
        },
        # R5: Eligible(x, AI) => CanRequestAILabSupport(x)
        {
            "name"      : "R5_EligibleCanRequest",
            "condition" : ("Eligible", "?x", "AI"),
            "conclusion": ("CanRequestAILabSupport", "?x")
        },
        # R6: UsesLab(x, AI_Lab) => CanRequestAILabSupport(x)
        {
            "name"      : "R6_UsesLabCanRequest",
            "condition" : ("UsesLab", "?x", "AI_Lab"),
            "conclusion": ("CanRequestAILabSupport", "?x")
        },
        # R7: Role(x, staff) => CanRequestMaintenance(x)
        {
            "name"      : "R7_StaffMaintenance",
            "condition" : ("Role", "?x", "staff"),
            "conclusion": ("CanRequestMaintenance", "?x")
        },
        # R8: Teaches(x, SE) => Instructor(x, SE)
        {
            "name"      : "R8_TeachesSE",
            "condition" : ("Teaches", "?x", "SE"),
            "conclusion": ("Instructor", "?x", "SE")
        },
    ]
}
 
 
def _match(pattern, fact):
   
    if len(pattern) != len(fact):
        return None
    bindings = {}
    for p, f in zip(pattern, fact):
        if isinstance(p, str) and p.startswith("?"):
            var = p[1:]
            if var in bindings and bindings[var] != f:
                return None
            bindings[var] = f
        elif p != f:
            return None
    return bindings
 
 
def _apply_bindings(template, bindings):
    return tuple(
        bindings.get(t[1:], t) if (isinstance(t, str) and t.startswith("?")) else t
        for t in template
    )
 
 
def forward_chain(kb):

    facts   = set(tuple(f) for f in kb["facts"])
    rules   = kb["rules"]
    steps   = []
    changed = True
 
    while changed:
        changed = False
        for rule in rules:
            cond  = tuple(rule["condition"])
            concl = tuple(rule["conclusion"])
 
            for fact in list(facts):
                bindings = _match(cond, fact)
                if bindings is not None:
                    new_fact = _apply_bindings(concl, bindings)
                    if new_fact not in facts:
                        facts.add(new_fact)
                        steps.append({
                            "rule"   : rule["name"],
                            "from"   : fact,
                            "derived": new_fact
                        })
                        changed = True
 
    return facts, steps
 
 
def check_eligibility(request_obj, kb=KNOWLEDGE_BASE):
    
    facts, steps = forward_chain(kb)
 
    name     = request_obj["name"]
    role     = request_obj["role"]
    category = request_obj["category"]
    rt       = request_obj["request_type"]
 
    result = dict(LOGIC_OUTPUT_TEMPLATE)
 
    # ── Eligibility_Check: evaluate user query directly 
    if rt == "Eligibility_Check":
        query = request_obj.get("query", "")
        for fact in facts:
            fact_str = f"{fact[0]}({', '.join(str(a) for a in fact[1:])})"
            if query.replace(" ", "") in fact_str.replace(" ", ""):
                result.update({
                    "allowed"    : True,
                    "entailed"   : True,
                    "explanation": f"Query '{query}' is entailed by the KB via forward chaining."
                })
                return result
        result.update({
            "allowed"    : False,
            "entailed"   : False,
            "explanation": f"Query '{query}' could NOT be entailed from the knowledge base."
        })
        return result
 
    #  Role-based permission checks 
    if category == "AI_Lab_Support":
        if ("CanRequestAILabSupport", name) in facts:
            result.update({
                "allowed"    : True,
                "entailed"   : True,
                "explanation": f"{name} is enrolled in AI / has lab access → eligible for AI Lab Support."
            })
        elif ("UsesLab", name, "AI_Lab") in facts:
            result.update({
                "allowed"    : True,
                "entailed"   : True,
                "explanation": f"{name} is an AI instructor → authorized to use AI_Lab."
            })
        else:
            result.update({
                "allowed"    : False,
                "entailed"   : False,
                "explanation": f"{name} is not enrolled in AI and does not have lab access. Request denied."
            })
 
    elif category == "Viva_Scheduling":
        if ("Enrolled", name, "AI") in facts or ("Enrolled", name, "SE") in facts:
            result.update({
                "allowed"    : True,
                "entailed"   : True,
                "explanation": f"{name} is enrolled in a course → eligible for viva scheduling."
            })
        elif ("Instructor", name, "AI") in facts or ("Instructor", name, "SE") in facts:
            result.update({
                "allowed"    : True,
                "entailed"   : True,
                "explanation": f"{name} is an instructor → authorized for viva scheduling."
            })
        else:
            result.update({
                "allowed"    : False,
                "entailed"   : False,
                "explanation": f"{name} is not enrolled in any course → viva scheduling denied."
            })
 
    elif category == "Maintenance":
        if ("CanRequestMaintenance", name) in facts:
            result.update({
                "allowed"    : True,
                "entailed"   : True,
                "explanation": f"{name} is staff → authorized to request maintenance."
            })
        else:
            result.update({
                "allowed"    : False,
                "entailed"   : False,
                "explanation": f"Only staff may request maintenance. {name} is not authorized."
            })
 
    elif category in ("Access_Request", "Emergency_Help"):
        # Any registered user (with a Role fact) may request these
        if ("Role", name, role) in facts:
            result.update({
                "allowed"    : True,
                "entailed"   : True,
                "explanation": f"{name} has role '{role}' → authorized for {category}."
            })
        else:
            result.update({
                "allowed"    : False,
                "entailed"   : False,
                "explanation": f"{name} is not recognized in the knowledge base."
            })
    else:
        result.update({
            "allowed"    : False,
            "entailed"   : False,
            "explanation": f"Category '{category}' is not handled by the KB."
        })
 
    return result
 
 
print("FOL Rules: R1 (Eligibility), R2 (Teaches→Instructor), "
      "R3 (Enrolled→UsesLab), R4 (Instructor→UsesLab)")
 
#  Test: Logic/KB 
print("\n Test: Module 2A — Logic/KB Module ")
logic_out = check_eligibility(req_obj)
print("Logic/KB Output for Ali (AI_Lab_Support):")
for k, v in logic_out.items():
    print(f"  {k:15s}: {v}")
 
print()
# Eligibility_Check test: DrKhan
ec_raw = {
    "name"        : "DrKhan",
    "role"        : "instructor",
    "request_type": "Eligibility_Check",
    "query"       : "UsesLab(DrKhan, AI_Lab)"
}
ec_req, ec_flags, ec_err = preprocess_request(ec_raw)
logic_out2 = check_eligibility(ec_req)
print("Logic/KB Output for DrKhan (Eligibility_Check):")
for k, v in logic_out2.items():
    print(f"  {k:15s}: {v}")
 
 

# CSP Scheduler
# Viva Groups (G1–G6) constraints:
#   G1 != G2  (slot clash — red)
#   G1 != G4  (slot clash — red)
#   G1 != G3  (examiner clash — purple)
#   G2 != G4  (slot clash — red)
#   G2 != G5  (supervisor clash — orange)
#   G3 != G5  (slot clash — red)
#   G3 != G6  (slot clash — red)
#   G4  < G3  (precedence — blue: G4 must be BEFORE G3)
#   G5 != G6  (slot clash — red)
 
VIVA_GROUPS = {
    "G1": {"examiner": "E1", "supervisor": "S1"},
    "G2": {"examiner": "E2", "supervisor": "S1"},
    "G3": {"examiner": "E1", "supervisor": "S2"},
    "G4": {"examiner": "E3", "supervisor": "S3"},
    "G5": {"examiner": "E2", "supervisor": "S1"},
    "G6": {"examiner": "E4", "supervisor": "S2"},
}
 
# Constraints from CSP diagram
VIVA_CONSTRAINTS = [
    ("G1", "G2", "slot_clash"),        # red edge
    ("G1", "G4", "slot_clash"),        # red edge
    ("G1", "G3", "examiner_clash"),    # purple edge
    ("G2", "G4", "slot_clash"),        # red edge
    ("G2", "G5", "supervisor_clash"),  # orange edge
    ("G3", "G5", "slot_clash"),        # red edge
    ("G3", "G6", "slot_clash"),        # red edge
    ("G4", "G3", "precedence"),        # blue arrow: G4 slot < G3 slot
    ("G5", "G6", "slot_clash"),        # red edge
]
 
TOTAL_SLOTS = [1, 2, 3, 4]
 
ROOMS_BY_CATEGORY = {
    "AI_Lab_Support"  : ["AI_Lab"],
    "Viva_Scheduling" : ["Exam_Hall", "Seminar_Room"],
    "Access_Request"  : ["Admin_Block", "Student_Services"],
    "Maintenance"     : ["Admin_Block"],
    "Emergency_Help"  : ["Medical_Center", "Admin_Block"]
}
 
# Simulated existing bookings
EXISTING_BOOKINGS = {
    ("AI_Lab",       1): "Group_A",
    ("Exam_Hall",    2): "Group_B",
    ("Seminar_Room", 1): "DrAli",
}
 
 
def _viva_consistent(group, slot, assignment, constraints):
 
    for (g1, g2, ctype) in constraints:
        if ctype == "precedence":
            # G4 < G3: G4 slot must be strictly less than G3 slot
            if g1 == group and g2 in assignment:
                if slot >= assignment[g2]:
                    return False, f"{g1} must have a lower slot than {g2} (precedence)"
            if g2 == group and g1 in assignment:
                if assignment[g1] >= slot:
                    return False, f"{g1} must have a lower slot than {g2} (precedence)"
        else:
            # All other constraints: conflicting groups must have different slots
            other_slot = None
            if g1 == group and g2 in assignment:
                other_slot = assignment[g2]
            elif g2 == group and g1 in assignment:
                other_slot = assignment[g1]
            if other_slot is not None and other_slot == slot:
                return False, f"{g1} and {g2} clash ({ctype})"
    return True, ""
 
 
def _backtrack_viva(groups, assignment, constraints, slots):
    if len(assignment) == len(groups):
        return assignment, "success"
 
    # Select next unassigned group
    unassigned = [g for g in groups if g not in assignment]
    group = unassigned[0]
 
    for slot in slots:
        ok, reason = _viva_consistent(group, slot, assignment, constraints)
        if ok:
            assignment[group] = slot
            result, msg = _backtrack_viva(groups, assignment, constraints, slots)
            if result is not None:
                return result, msg
            del assignment[group]  # backtrack
 
    return None, f"No valid slot found for {group}"
 
 
def run_viva_csp(group_ids=None):
  
    groups   = group_ids if group_ids else list(VIVA_GROUPS.keys())
    schedule, msg = _backtrack_viva(groups, {}, VIVA_CONSTRAINTS, TOTAL_SLOTS)
    return schedule, msg
 
 
def _slot_free(room, slot, bookings):
    return (room, slot) not in bookings
 
 
def run_general_csp(request_obj, logic_result, bookings=None):
    
    if bookings is None:
        bookings = dict(EXISTING_BOOKINGS)
 
    result = dict(CSP_OUTPUT_TEMPLATE)
 
    if not logic_result.get("allowed", False):
        result.update({"decision": "rejected", "notes": "Rejected by Logic/KB."})
        return result
 
    category  = request_obj.get("category", "")
    pref_slot = request_obj.get("preferred_slot")
    name      = request_obj.get("name", "User")
    rooms     = ROOMS_BY_CATEGORY.get(category, [])
 
    if not rooms:
        result.update({"decision": "rejected",
                        "notes": f"No rooms defined for category '{category}'."})
        return result
 
    # Preferred slot first, then try remaining slots
    if pref_slot and int(pref_slot) in TOTAL_SLOTS:
        slots = [int(pref_slot)] + [s for s in TOTAL_SLOTS if s != int(pref_slot)]
    else:
        slots = list(TOTAL_SLOTS)
 
    for room in rooms:
        for slot in slots:
            if _slot_free(room, slot, bookings):
                bookings[(room, slot)] = name
                note = f"Assigned {room} slot {slot}."
                if pref_slot and slot != int(pref_slot):
                    note += f" (Preferred slot {pref_slot} was unavailable.)"
                result.update({
                    "decision"      : "accepted",
                    "assigned_room" : room,
                    "assigned_slot" : slot,
                    "destination"   : room,
                    "notes"         : note
                })
                return result
 
    result.update({"decision": "rejected", "notes": "No available room-slot combination found"})
    return result
 
 
def run_csp_scheduler(request_obj, logic_result, ann_priority=None, bookings=None):

    result = dict(CSP_OUTPUT_TEMPLATE)
 
    if not logic_result.get("allowed", False):
        result.update({"decision": "rejected",
                        "notes": "Rejected by Logic/KB — CSP not executed"})
        return result
 
    category = request_obj.get("category", "")
 
    if category == "Viva_Scheduling":
        group_id = request_obj.get("group_id", "")
        schedule, msg = run_viva_csp()
        if schedule:
            target_group  = group_id if group_id in schedule else list(schedule.keys())[0]
            assigned_slot = schedule.get(target_group)
            room          = "Exam_Hall"
            result.update({
                "decision"      : "accepted",
                "assigned_room" : room,
                "assigned_slot" : assigned_slot,
                "destination"   : room,
                "notes"         : (f"Viva CSP schedule: {schedule}. "
                                   f"G4(slot {schedule['G4']}) < G3(slot {schedule['G3']}) — "
                                   f"Precedence constraint satisfied.")
            })
        else:
            result.update({"decision": "rejected", "notes": msg})
        return result
    else:
        return run_general_csp(request_obj, logic_result, bookings)
 
 

schedule, msg = run_viva_csp()
if schedule:
    print(f"{'Group':<8} {'Slot':<6}")
    print(" " * 16)
    for g, s in sorted(schedule.items()):
        print(f"  {g}  →  Slot {s}")
    prec_ok = schedule['G4'] < schedule['G3']
    print(f"\nPrecedence G4 < G3: G4=Slot{schedule['G4']}, G3=Slot{schedule['G3']} "
          f"→ {' SATISFIED' if prec_ok else ' VIOLATED'}")
else:
    print("Viva CSP failed:", msg)
 
print("\n Test: Module 2B — CSP Scheduler ")
csp_out = run_csp_scheduler(req_obj, logic_out, ann_priority=None)
print("CSP Output:")
for k, v in csp_out.items():
    print(f"  {k:20s}: {v}")

# Distances from the weighted campus graph
### Campus distances : Manhattan from coordinates in diagram 
CAMPUS_DISTANCES = {
    ("Hostel",        "AI_Lab")  : 9,   # (2,0) -> |9-2|+|2-0| = 9
    ("Hostel",        "Exam_Hall"): 13, # (2,0) -> |8-2|+|5-0| = 11
    ("Main_Gate",     "AI_Lab")  : 11,  # (0,4) -> 9+2 = 11
    ("Main_Gate",     "Exam_Hall"): 12, # (0,4) -> 8+1 = 9
    ("Parking",       "AI_Lab")  : 9,   # (2,4) -> 7+2 = 9
    ("Bus_Stop",      "AI_Lab")  : 10,  # (0,1) -> 9+1 = 10
    ("Medical_Center","AI_Lab")  : 9,   # (1,1) -> 8+1 = 9
    ("Cafeteria",     "AI_Lab")  : 6,   # (4,1) -> 5+1 = 6
    ("Library",       "AI_Lab")  : 3,   # (6,2) -> 3+0 = 3
    ("Science_Block", "AI_Lab")  : 3,   # (7,1) -> 2+1 = 3
    ("Admin_Block",   "AI_Lab")  : 9,   # (3,5) -> 6+3 = 9
    ("Exam_Hall",     "AI_Lab")  : 4,   # (8,5) -> 1+3 = 4
    ("Seminar_Room",  "AI_Lab")  : 3,   # (10,4)-> 1+2 = 3
}
def get_distance(src, dst):
    key = (src, dst)
    rev = (dst, src)
    if key in CAMPUS_DISTANCES:
        return CAMPUS_DISTANCES[key]
    if rev in CAMPUS_DISTANCES:
        return CAMPUS_DISTANCES[rev]
    return 5   # default if unknown pair
 
 
def build_feature_vector(request_obj, eligibility_result=True):

    role_enc  = ROLE_ENCODING.get(request_obj["role"], 0)
    cat_enc   = REQUEST_TYPE_ENCODING.get(request_obj["category"], 0)
    severity  = request_obj.get("severity", 5)
    time_sens = request_obj.get("time_sensitivity", 5)
    crowd     = request_obj.get("crowd_level", 5)
    distance  = get_distance(
                    request_obj.get("current_location", ""),
                    request_obj.get("destination", "AI_Lab")
                )
    elig      = encode_bool(eligibility_result)
 
    # Normalize to [0,1] for better ANN training
    return [
        role_enc / 2.0,
        cat_enc / 4.0,
        severity / 10.0,
        time_sens / 10.0,
        crowd / 10.0,
        distance / 13.0,
        float(elig)
    ]
 
 
fv_demo = build_feature_vector(req_obj)
print(f"Feature Vector for Ali's request:")
for name, val in zip(FEATURE_ORDER, fv_demo):
    print(f"  {name:20s}: {val}")
 
 
# Perceptron: Binary Classifier
# Output: 0 = not_urgent,  1 = urgent
 
class Perceptron:
 
 
    def __init__(self, n_features=7, learning_rate=0.1, epochs=200):
        self.lr      = learning_rate
        self.epochs  = epochs
        self.weights = [0.0] * n_features
        self.bias    = 0.0
 
    def _activation(self, net_input):
        """Step activation function: output 1 if net >= 0.5 else 0."""
        return 1 if net_input >= 0.5 else 0
 
    def _net_input(self, x):
        """Weighted sum: Σ(wi * xi) + bias."""
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
 
    def predict_single(self, x):
        return self._activation(self._net_input(x))
 
    def predict(self, X):
        return [self.predict_single(x) for x in X]
 
    def train(self, X, y):
        """
        Train using Perceptron learning rule:
          w = w + lr * (target - output) * x
          b = b + lr * (target - output)
        """
        history = []
        for epoch in range(self.epochs):
            total_error = 0
            for xi, yi in zip(X, y):
                pred  = self.predict_single(xi)
                error = yi - pred
                total_error += abs(error)
                self.weights = [
                    w + self.lr * error * xi_j
                    for w, xi_j in zip(self.weights, xi)
                ]
                self.bias += self.lr * error
            history.append(total_error)
            if total_error == 0:
                print(f"  Perceptron converged at epoch {epoch + 1}")
                break
        return history
 
    def accuracy(self, X, y):
        preds   = self.predict(X)
        correct = sum(p == t for p, t in zip(preds, y))
        return correct / len(y)
 
    def predict_label(self, x):
        p = self.predict_single(x)
        return "urgent" if p == 1 else "not_urgent"
 
 

 
# Perceptron Training Data
# Features: [Role, RequestType, Severity, TimeSensitivity, CrowdLevel, Distance, Eligibility]
# Labels:    1 = urgent,  0 = not_urgent
 
PERCEPTRON_TRAIN_X = [
    # Urgent samples (label = 1)
    [0, 0, 9, 9, 8, 3, 1],   # student, AI_Lab, very high severity
    [0, 0, 8, 8, 7, 4, 1],   # student, AI_Lab, high severity
    [1, 1, 7, 8, 6, 2, 1],   # instructor, Viva, high
    [0, 4, 9, 9, 9, 5, 1],   # student, Emergency, extreme
    [0, 0, 7, 8, 5, 4, 1],   # student, AI_Lab, moderate-high
    [0, 2, 9, 9, 8, 6, 1],   # student, Access, emergency level
    [1, 1, 8, 9, 7, 2, 1],   # instructor, Viva urgent
    [0, 0, 6, 7, 5, 4, 1],   # student, AI_Lab, moderate
    # Not urgent samples (label = 0)
    [0, 3, 4, 3, 2, 5, 1],   # student, Maintenance, low
    [1, 0, 3, 2, 2, 3, 1],   # instructor, AI_Lab, low
    [2, 3, 2, 2, 1, 4, 1],   # staff, Maintenance, very low
    [0, 2, 2, 2, 1, 7, 0],   # student, Access, low + not eligible
    [1, 0, 1, 1, 1, 2, 1],   # instructor, AI_Lab, trivial
    [0, 0, 3, 3, 2, 4, 1],   # student, AI_Lab, low severity
    [2, 3, 4, 3, 3, 3, 1],   # staff, Maintenance, moderate
    [0, 1, 2, 2, 2, 5, 0],   # student, Viva, low + not eligible
]
 
PERCEPTRON_TRAIN_Y = [1, 1, 1, 1, 1, 1, 1, 1,
                      0, 0, 0, 0, 0, 0, 0, 0]
 
print("\nTraining Perceptron")
perceptron = Perceptron(n_features=7, learning_rate=0.1, epochs=200)
p_history  = perceptron.train(PERCEPTRON_TRAIN_X, PERCEPTRON_TRAIN_Y)
 
train_acc  = perceptron.accuracy(PERCEPTRON_TRAIN_X, PERCEPTRON_TRAIN_Y)
print(f"Perceptron training accuracy: {train_acc * 100:.1f}%")
print(f"Final weights: {[round(w, 3) for w in perceptron.weights]}")
print(f"Final bias:    {round(perceptron.bias, 3)}")
 
 
# Multiclass Classifier
#   Input Layer  : 7 features (x1-x7)
#   Hidden Layer 1: 4 nodes  (h1_1, h1_2, h1_3, h1_4)
#   Hidden Layer 2: 3 nodes  (h2_1, h2_2, h2_3)
#   Output Layer : 4 classes (Low, Normal, High, Urgent)
#   Activation   : Sigmoid (hidden), Softmax (output)
 
class MLP:
  
 
    LABELS = {0: "low", 1: "normal", 2: "high", 3: "urgent"}
 
    def __init__(self, input_size=7, hidden1=8, hidden2=6, output_size=4,
                 learning_rate=0.1, epochs=1000):
        self.lr          = learning_rate
        self.epochs      = epochs
        self.input_size  = input_size
        self.hidden1     = hidden1
        self.hidden2     = hidden2
        self.output_size = output_size
        self._init_weights()
 
    def _init_weights(self):
        random.seed(42)
 
        def rand_w(rows, cols):
            scale = math.sqrt(2.0 / cols)
            return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]
 
        self.W1 = rand_w(self.hidden1, self.input_size)   # 4 x 7
        self.b1 = [0.0] * self.hidden1
        self.W2 = rand_w(self.hidden2, self.hidden1)       # 3 x 4
        self.b2 = [0.0] * self.hidden2
        self.W3 = rand_w(self.output_size, self.hidden2)   # 4 x 3
        self.b3 = [0.0] * self.output_size
 
    def _sigmoid(self, x):
        """Sigmoid activation: 1 / (1 + e^-x) (clamped for stability)."""
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))
 
    def _sigmoid_deriv(self, s):
        """Derivative of sigmoid given sigmoid output s: s*(1-s)."""
        return s * (1.0 - s)
 
    def _softmax(self, vec):
        """Softmax over output vector."""
        max_v = max(vec)
        exps  = [math.exp(v - max_v) for v in vec]
        s     = sum(exps)
        return [e / s for e in exps]
 
    def _dot(self, W_row, x, b):
        return sum(w * xi for w, xi in zip(W_row, x)) + b
 
    # Forward Pass 
    def _forward(self, x):
        h1 = [self._sigmoid(self._dot(self.W1[i], x, self.b1[i]))
              for i in range(self.hidden1)]
        h2 = [self._sigmoid(self._dot(self.W2[i], h1, self.b2[i]))
              for i in range(self.hidden2)]
        out_raw = [self._dot(self.W3[i], h2, self.b3[i])
                   for i in range(self.output_size)]
        out = self._softmax(out_raw)
        return h1, h2, out
 
    # Backpropagation
    def _backprop(self, x, y_true):
        h1, h2, out = self._forward(x)
 
        target = [1.0 if i == y_true else 0.0 for i in range(self.output_size)]
 
        delta3 = [out[i] - target[i] for i in range(self.output_size)]
 
        delta2 = []
        for j in range(self.hidden2):
            err = sum(self.W3[k][j] * delta3[k] for k in range(self.output_size))
            delta2.append(err * self._sigmoid_deriv(h2[j]))
 
        delta1 = []
        for j in range(self.hidden1):
            err = sum(self.W2[k][j] * delta2[k] for k in range(self.hidden2))
            delta1.append(err * self._sigmoid_deriv(h1[j]))
 
        for i in range(self.output_size):
            for j in range(self.hidden2):
                self.W3[i][j] -= self.lr * delta3[i] * h2[j]
            self.b3[i] -= self.lr * delta3[i]
 
        for i in range(self.hidden2):
            for j in range(self.hidden1):
                self.W2[i][j] -= self.lr * delta2[i] * h1[j]
            self.b2[i] -= self.lr * delta2[i]
 
        for i in range(self.hidden1):
            for j in range(self.input_size):
                self.W1[i][j] -= self.lr * delta1[i] * x[j]
            self.b1[i] -= self.lr * delta1[i]
 
        loss = -math.log(max(out[y_true], 1e-9))
        return loss
 
    def train(self, X, y):
        history = []
        for epoch in range(self.epochs):
            total_loss = sum(self._backprop(xi, yi) for xi, yi in zip(X, y))
            history.append(total_loss)
            if (epoch + 1) % 100 == 0:
                acc = self.accuracy(X, y)
                print(f"  Epoch {epoch+1:4d}  loss={total_loss:.4f}  acc={acc*100:.1f}%")
        return history
 
    def predict_single(self, x):
        _, _, out = self._forward(x)
        return out.index(max(out))
 
    def predict(self, X):
        return [self.predict_single(x) for x in X]
 
    def predict_proba(self, x):
        _, _, out = self._forward(x)
        return out
 
    def predict_label(self, x):
        proba     = self.predict_proba(x)
        class_idx = proba.index(max(proba))
        confidence = max(proba)
        return self.LABELS[class_idx], round(confidence, 4)
 
    def accuracy(self, X, y):
        preds   = self.predict(X)
        correct = sum(p == t for p, t in zip(preds, y))
        return correct / len(y)
 
 
 
#  MLP Training Data 
# Features: [Role, RequestType, Severity, TimeSensitivity, CrowdLevel, Distance, Eligibility]
# Labels:    0=low, 1=normal, 2=high, 3=urgent
 
MLP_TRAIN_X = [
    # Urgent (label = 3) — normalized [role/2, cat/4, sev/10, time/10, crowd/10, dist/13, elig]
    [0.0,  0.0,  0.9,  0.9,  0.9,  0.2308, 1.0],
    [0.0,  1.0,  0.9,  0.9,  0.9,  0.3846, 1.0],
    [0.5,  0.25, 0.9,  0.9,  0.8,  0.1538, 1.0],
    [0.0,  0.0,  0.8,  0.9,  0.8,  0.3077, 1.0],
    [0.0,  0.5,  0.9,  0.9,  0.9,  0.4615, 1.0],
    [0.0,  1.0,  1.0,  1.0,  0.9,  0.3077, 1.0],
    [0.5,  0.0,  0.9,  1.0,  0.9,  0.2308, 1.0],
    [0.0,  0.5,  1.0,  0.9,  0.8,  0.3846, 1.0],
    # High (label = 2)
    [0.0,  0.0,  0.8,  0.8,  0.7,  0.3077, 1.0],
    [0.5,  0.25, 0.8,  0.8,  0.6,  0.1538, 1.0],
    [0.0,  0.0,  0.7,  0.8,  0.7,  0.2308, 1.0],
    [0.0,  0.5,  0.8,  0.8,  0.6,  0.3846, 1.0],
    [0.5,  0.0,  0.7,  0.7,  0.6,  0.2308, 1.0],
    [0.0,  0.25, 0.8,  0.7,  0.7,  0.3077, 1.0],
    [1.0,  0.0,  0.7,  0.8,  0.6,  0.2308, 1.0],
    [0.0,  0.75, 0.8,  0.7,  0.7,  0.3846, 1.0],
    # Normal (label = 1)
    [0.0,  0.0,  0.5,  0.5,  0.5,  0.3077, 1.0],
    [0.5,  0.25, 0.5,  0.6,  0.4,  0.2308, 1.0],
    [1.0,  0.75, 0.5,  0.4,  0.4,  0.3077, 1.0],
    [0.0,  0.5,  0.4,  0.5,  0.3,  0.3846, 1.0],
    [0.5,  0.0,  0.6,  0.5,  0.5,  0.1538, 1.0],
    [0.0,  0.25, 0.5,  0.5,  0.4,  0.3077, 1.0],
    [1.0,  0.5,  0.6,  0.5,  0.5,  0.2308, 1.0],
    [0.5,  0.75, 0.4,  0.6,  0.4,  0.3846, 1.0],
    # Low (label = 0)
    [0.0,  0.75, 0.2,  0.2,  0.1,  0.3846, 1.0],
    [0.5,  0.0,  0.1,  0.1,  0.1,  0.1538, 1.0],
    [1.0,  0.75, 0.2,  0.2,  0.2,  0.2308, 1.0],
    [0.0,  0.25, 0.2,  0.2,  0.1,  0.4615, 0.0],
    [0.0,  0.0,  0.3,  0.2,  0.2,  0.3077, 0.0],
    [0.5,  0.75, 0.1,  0.1,  0.1,  0.2308, 1.0],
    [1.0,  0.0,  0.2,  0.1,  0.1,  0.3077, 1.0],
    [0.0,  0.5,  0.3,  0.2,  0.2,  0.3846, 0.0],
]

MLP_TRAIN_Y = [3, 3, 3, 3, 3, 3, 3, 3,   # urgent
               2, 2, 2, 2, 2, 2, 2, 2,   # high
               1, 1, 1, 1, 1, 1, 1, 1,   # normal
               0, 0, 0, 0, 0, 0, 0, 0]   # low

print("\nTraining MLP (1000 epochs)")
mlp = MLP(input_size=7, hidden1=8, hidden2=6, output_size=4,
          learning_rate=0.1, epochs=1000)
mlp_history = mlp.train(MLP_TRAIN_X, MLP_TRAIN_Y)
 
final_acc = mlp.accuracy(MLP_TRAIN_X, MLP_TRAIN_Y)
print(f"\nFinal MLP training accuracy: {final_acc * 100:.1f}%")
 
 
 
def run_ann_module(request_obj, eligibility_hint=True):
    
    fv = build_feature_vector(request_obj, eligibility_hint)
 
    binary_label = perceptron.predict_label(fv)
 
    final_label, confidence = mlp.predict_label(fv)
 
    priority_output = dict(PRIORITY_OUTPUT_TEMPLATE)
    priority_output.update({
        "binary_priority": binary_label,
        "final_priority" : final_label,
        "confidence"     : confidence
    })
    return priority_output, fv
 
 
 
print("\nTest: Module 3 — ANN Module ")
ann_out, fv_used = run_ann_module(req_obj)
print(f"Feature Vector    : {fv_used}")
print(f"Perceptron output : {ann_out['binary_priority']}  (binary baseline)")
print(f"MLP output        : {ann_out['final_priority']}  (multiclass priority)")
print(f"Confidence        : {ann_out['confidence']}")
 

def build_final_response(request_obj, router_out,
                          ann_out=None, logic_out=None,
                          csp_out=None, search_out=None):
  
    response = dict(FINAL_RESPONSE_TEMPLATE)
    response["request_id"] = request_obj["request_id"]
    rt = request_obj["request_type"]
 
    if logic_out and not logic_out.get("allowed", True) and rt != "Navigation_Only":
        response["decision"] = "rejected"
    elif csp_out and csp_out.get("decision") == "rejected":
        response["decision"] = "rejected"
    elif search_out and not search_out.get("path"):
        response["decision"] = "no_route"
    else:
        response["decision"] = "accepted" if csp_out else "completed"
 
    if ann_out:
        response["priority"] = {
            "binary_priority": ann_out["binary_priority"],
            "final_priority" : ann_out["final_priority"],
            "confidence"     : ann_out["confidence"]
        }
 
    if logic_out:
        response["eligibility"] = {
            "allowed"    : logic_out.get("allowed", False),
            "entailed"   : logic_out.get("entailed", False),
            "explanation": logic_out.get("explanation", "")
        }
 
    if csp_out and csp_out.get("decision") == "accepted":
        response["assignment"] = {
            "room" : csp_out["assigned_room"],
            "slot" : csp_out["assigned_slot"],
            "notes": csp_out["notes"]
        }
 
    if search_out and search_out.get("path"):
        response["route"] = {
            "algorithm": search_out["algorithm_used"],
            "path"     : search_out["path"],
            "cost"     : search_out["cost"],
            "steps"    : search_out["steps"]
        }
 
    name = request_obj["name"]
 
    if response["decision"] == "rejected":
        reason = ""
        if logic_out and not logic_out.get("allowed"):
            reason = logic_out.get("explanation", "eligibility check failed")
        elif csp_out:
            reason = csp_out.get("notes", "scheduling conflict")
        response["message"] = f"Request rejected for {name}. Reason: {reason}"
 
    elif response["decision"] == "no_route":
        response["message"] = (
            f"No valid route found from {request_obj['current_location']} "
            f"to {request_obj.get('destination', 'destination')}."
        )
 
    elif rt == "Navigation_Only":
        p = " → ".join(search_out["path"])
        response["message"] = (
            f"Best route for {name}: {p}. "
            f"Cost: {search_out['cost']}, Steps: {search_out['steps']}."
        )
 
    elif rt == "Eligibility_Check":
        status = "ENTAILED" if logic_out.get("entailed") else "NOT entailed"
        response["message"] = (
            f"Eligibility query for {name}: {status}. "
            f"{logic_out.get('explanation', '')}"
        )
 
    elif rt in ("Booking_or_Scheduling", "Urgent_Service_Request"):
        response["message"] = (
            f"Request accepted for {name}. "
            f"Assigned {csp_out['assigned_room']} in slot {csp_out['assigned_slot']}."
        )
 
    elif rt == "Full_Service_Request":
        path_str = (" → ".join(search_out["path"])
                    if search_out and search_out.get("path") else "N/A")
        response["message"] = (
            f"Full service accepted for {name} "
            f"(Priority: {ann_out['final_priority']}, "
            f"Confidence: {ann_out['confidence']}). "
            f"Assigned {csp_out['assigned_room']} slot {csp_out['assigned_slot']}. "
            f"Route: {path_str}."
        )
    else:
        response["message"] = f"Request for {name} processed successfully."
 
    return response
 
 
def print_response(response):
    """Pretty-print the final response."""
    print(" " * 65)
    print("FINAL RESPONSE")
    print(" " * 65)
    print(f"  Request ID : {response['request_id']}")
    print(f"  Decision   : {response['decision'].upper()}")
 
    if response.get("priority"):
        p = response["priority"]
        print(f"  Priority   : {p['final_priority'].upper()} "
              f"(binary: {p['binary_priority']}, confidence: {p['confidence']})")
 
    if response.get("eligibility"):
        e = response["eligibility"]
        print(f"  Eligibility: {'ALLOWED ' if e['allowed'] else 'DENIED '}")
        print(f"               {e['explanation']}")
 
    if response.get("assignment"):
        a = response["assignment"]
        print(f"  Assignment : Room={a['room']}, Slot={a['slot']}")
        print(f"               {a['notes']}")
 
    if response.get("route"):
        r = response["route"]
        print(f"  Route      : {' → '.join(r['path'])}")
        print(f"               Algorithm: {r['algorithm']}, "
              f"Cost: {r['cost']}, Steps: {r['steps']}")
 
    print(f"  Message    : {response['message']}")
    print(" " * 65)
 
 

 
def run_pipeline(raw_input):
    """
    Complete end-to-end Smart Campus AI pipeline:
      Preprocess → Route → ANN (M3) → Logic/KB (M2A) → CSP (M2B) → Search (M1) → Response
    """
    print("\n" + " " * 65)
    print("SMART CAMPUS AI  PIPELINE START")
    print(" " * 65)
 
    # Step 1: Preprocessing
    print("\n[Step 1] Input & Preprocessing")
    req_obj, p_flags, errors = preprocess_request(raw_input)
    if errors:
        print("  VALIDATION ERRORS:", errors)
        return None
    print(f"  Request ID : {req_obj['request_id']}")
    print(f"  Name       : {req_obj['name']}  |  Role: {req_obj['role']}")
    print(f"  Type       : {req_obj['request_type']}")
    print(f"  Pipeline   : ANN={p_flags['needs_ann']}, Logic={p_flags['needs_logic']}, "
          f"CSP={p_flags['needs_csp']}, Search={p_flags['needs_search']}")
 
    # Step 2: Router
    print("\n[Step 2] Request Router")
    r_out, err = route_request(req_obj, p_flags)
    if err:
        print("  Router Error:", err)
        return None
    print(f"  Pipeline selected: {r_out['selected_pipeline']}")
 
    ann_out    = None
    logic_out  = None
    csp_out    = None
    search_out = None
 
    if p_flags["needs_ann"]:
        print("\n[Step 3] Module 3 (ANN) — Student 3: Priority Prediction")
        ann_out, _ = run_ann_module(req_obj)
        print(f"  Perceptron : {ann_out['binary_priority']}  (binary)")
        print(f"  MLP        : {ann_out['final_priority'].upper()}  "
              f"(confidence: {ann_out['confidence']})")
 
    if p_flags["needs_logic"]:
        print("\n[Step 4] Module 2A (Logic/KB) — Student 2: Eligibility Check")
        logic_out = check_eligibility(req_obj)
        status    = "ALLOWED " if logic_out["allowed"] else "DENIED "
        print(f"  Status : {status}")
        print(f"  Reason : {logic_out['explanation']}")
 
        if not logic_out["allowed"] and req_obj["request_type"] != "Eligibility_Check":
            print("  → Request REJECTED at Logic/KB gate. Pipeline stops.")
            return build_final_response(req_obj, r_out, ann_out, logic_out, None, None)
 
    if p_flags["needs_csp"]:
        print("\n[Step 5] Module 2B (CSP) — Student 2: Room & Slot Scheduling")
        csp_out = run_csp_scheduler(req_obj, logic_out or {"allowed": True},
                                     ann_priority=ann_out)
        print(f"  Decision : {csp_out['decision'].upper()}")
        if csp_out["decision"] == "accepted":
            print(f"  Assigned : {csp_out['assigned_room']} — Slot {csp_out['assigned_slot']}")
        else:
            print(f"  Notes    : {csp_out['notes']}")
 
    if p_flags["needs_search"]:
        print("\n[Step 6] Module 1 (Search) — Student 1: Campus Navigation")
        src = req_obj["current_location"]
        dst = (csp_out["destination"]
               if csp_out and csp_out.get("destination")
               else req_obj.get("destination", ""))
        if src and dst:
            search_out = run_search_module(src, dst, graph_type="weighted")
            print(f"  Route    : {' → '.join(search_out['path'])}")
            print(f"  Algorithm: {search_out['algorithm_used']}, "
                  f"Cost: {search_out['cost']}, Steps: {search_out['steps']}")
        else:
            print("  Source or destination missing , skipping search ")
 
    print("\n[Step 7] Final Response Layer")
    response = build_final_response(req_obj, r_out,
                                     ann_out, logic_out,
                                     csp_out, search_out)
    print_response(response)
    return response
 
 
 
 
#
 
print("\n" + " " * 65)
print("SYSTEM TESTS  ALL 5 REQUEST TYPES")
print(" " * 65)
 
print("\n TEST 1: Navigation_Only ")
r1 = run_pipeline({
    "name"             : "Ali",
    "role"             : "student",
    "request_type"     : "Navigation_Only",
    "current_location" : "Hostel",
    "destination"      : "AI_Lab"
})
 
print("\n TEST 2: Eligibility_Check")
r2 = run_pipeline({
    "name"        : "DrKhan",
    "role"        : "instructor",
    "request_type": "Eligibility_Check",
    "query"       : "UsesLab(DrKhan, AI_Lab)"
})
 
print("\n TEST 3: Booking_or_Scheduling")
r3 = run_pipeline({
    "name"             : "Ali",
    "role"             : "student",
    "request_type"     : "Booking_or_Scheduling",
    "category"         : "AI_Lab_Support",
    "current_location" : "Hostel",
    "preferred_slot"   : 2
})
 
print("\n TEST 4: Urgent_Service_Request ")
r4 = run_pipeline({
    "name"             : "Hamza",
    "role"             : "student",
    "request_type"     : "Urgent_Service_Request",
    "category"         : "AI_Lab_Support",
    "current_location" : "Hostel",
    "severity"         : 8,
    "time_sensitivity" : 9,
    "crowd_level"      : 5,
    "preferred_slot"   : 2
})
 
print("\n TEST 5: Full_Service_Request")
r5 = run_pipeline({
    "name"             : "Ali",
    "role"             : "student",
    "request_type"     : "Full_Service_Request",
    "category"         : "AI_Lab_Support",
    "current_location" : "Hostel",
    "preferred_slot"   : 2,
    "severity"         : 8,
    "time_sensitivity" : 9,
    "crowd_level"      : 5,
    "description_note" : "Need urgent help before practical evaluation."
})
 
print("\n TEST 6: Rejected — Sara not enrolled in AI ")
r6 = run_pipeline({
    "name"             : "Sara",
    "role"             : "student",
    "request_type"     : "Full_Service_Request",
    "category"         : "AI_Lab_Support",
    "current_location" : "Hostel",
    "preferred_slot"   : 1,
    "severity"         : 7,
    "time_sensitivity" : 8,
    "crowd_level"      : 4
})
 
print("\n TEST 7: Viva Scheduling (Hamza, Group G1) ")
r7 = run_pipeline({
    "name"             : "Hamza",
    "role"             : "student",
    "request_type"     : "Booking_or_Scheduling",
    "category"         : "Viva_Scheduling",
    "current_location" : "Hostel",
    "preferred_slot"   : 1,
    "group_id"         : "G1"
})
 

def cli_interface():
 
    print()
    print(" " * 60)
    print("   SMART CAMPUS AI  DECISION SUPPORT SYSTEM")
    print(" " * 60)
 
    #  Name
    while True:
        name = input("\nEnter Name: ").strip()
        if name:
            break
        print("   Name cannot be empty. Please try again.")
 
    #  Role 
    while True:
        print("  Roles: student | instructor | staff")
        role = input("Enter Role: ").strip().lower()
        if role in VALID_ROLES:
            break
        print(f"   Invalid role '{role}'. Must be: student, instructor, or staff.")
 
    #  Request Type 
    while True:
        print("\nSelect Request Type:")
        print("  1. Navigation_Only")
        print("  2. Eligibility_Check")
        print("  3. Booking_or_Scheduling")
        print("  4. Urgent_Service_Request")
        print("  5. Full_Service_Request")
        choice = input("Enter choice (1-5): ").strip()
        rt_map = {
            "1": "Navigation_Only",
            "2": "Eligibility_Check",
            "3": "Booking_or_Scheduling",
            "4": "Urgent_Service_Request",
            "5": "Full_Service_Request"
        }
        if choice in rt_map:
            request_type = rt_map[choice]
            break
        print(f"   Invalid choice '{choice}'. Please enter 1 to 5.")
 
    raw = {"name": name, "role": role, "request_type": request_type}
 
    # Conditional Fields 
    if request_type == "Navigation_Only":
        print("\nKnown Locations:")
        print("  ", ", ".join(sorted(VALID_LOCATIONS)))
        while True:
            loc = input("Enter Current Location: ").strip()
            if loc in VALID_LOCATIONS:
                raw["current_location"] = loc
                break
            print(f"   Unknown location '{loc}'. Please choose from known locations.")
        while True:
            dst = input("Enter Destination     : ").strip()
            if dst in VALID_LOCATIONS:
                raw["destination"] = dst
                break
            print(f"   Unknown destination '{dst}'. Please choose from known locations.")
 
    elif request_type == "Eligibility_Check":
        print("\nExample: UsesLab(DrKhan, AI_Lab)  or  Eligible(Ali, AI)")
        while True:
            query = input("Enter Query: ").strip()
            if query:
                raw["query"] = query
                break
            print("   Query cannot be empty.")
 
    elif request_type == "Booking_or_Scheduling":
        print("\nCategories: AI_Lab_Support | Viva_Scheduling | Access_Request | Maintenance | Emergency_Help")
        while True:
            cat = input("Enter Category        : ").strip()
            if cat in VALID_CATEGORIES:
                raw["category"] = cat
                break
            print(f"   Invalid category '{cat}'.")
        print("\nKnown Locations:", ", ".join(sorted(VALID_LOCATIONS)))
        while True:
            loc = input("Enter Current Location: ").strip()
            if loc in VALID_LOCATIONS:
                raw["current_location"] = loc
                break
            print(f"   Unknown location '{loc}'.")
        while True:
            try:
                slot = int(input("Enter Preferred Slot (1-4): ").strip())
                if slot in VALID_SLOTS:
                    raw["preferred_slot"] = slot
                    break
                print("   Slot must be 1, 2, 3, or 4.")
            except ValueError:
                print("   Please enter a number (1-4).")
        if cat == "Viva_Scheduling":
            raw["group_id"] = input("Enter Group ID (e.g. G1): ").strip()
 
    elif request_type == "Urgent_Service_Request":
        print("\nCategories: AI_Lab_Support | Viva_Scheduling | Access_Request | Maintenance | Emergency_Help")
        while True:
            cat = input("Enter Category        : ").strip()
            if cat in VALID_CATEGORIES:
                raw["category"] = cat
                break
            print(f"   Invalid category '{cat}'.")
        print("\nKnown Locations:", ", ".join(sorted(VALID_LOCATIONS)))
        while True:
            loc = input("Enter Current Location: ").strip()
            if loc in VALID_LOCATIONS:
                raw["current_location"] = loc
                break
            print(f"   Unknown location '{loc}'.")
        for field, label in [("severity","Severity"),("time_sensitivity","Time Sensitivity"),("crowd_level","Crowd Level")]:
            while True:
                try:
                    val = int(input(f"Enter {label} (1-10)      : ").strip())
                    if 1 <= val <= 10:
                        raw[field] = val
                        break
                    print(f"  {label} must be between 1 and 10.")
                except ValueError:
                    print(f"  Please enter a number (1-10).")
        while True:
            try:
                slot = int(input("Enter Preferred Slot (1-4): ").strip())
                if slot in VALID_SLOTS:
                    raw["preferred_slot"] = slot
                    break
                print("   Slot must be 1, 2, 3, or 4")
            except ValueError:
                print("   Please enter a number (1-4)")
 
    elif request_type == "Full_Service_Request":
        print("\nCategories: AI_Lab_Support | Viva_Scheduling | Access_Request | Maintenance | Emergency_Help")
        while True:
            cat = input("Enter Category        : ").strip()
            if cat in VALID_CATEGORIES:
                raw["category"] = cat
                break
            print(f"   Invalid category '{cat}'.")
        print("\nKnown Locations:", ", ".join(sorted(VALID_LOCATIONS)))
        while True:
            loc = input("Enter Current Location: ").strip()
            if loc in VALID_LOCATIONS:
                raw["current_location"] = loc
                break
            print(f"   Unknown location '{loc}'.")
        while True:
            try:
                slot = int(input("Enter Preferred Slot (1-4)   : ").strip())
                if slot in VALID_SLOTS:
                    raw["preferred_slot"] = slot
                    break
                print("  Slot must be 1, 2, 3, or 4")
            except ValueError:
                print("  Please enter a number (1-4).")
        for field, label in [("severity","Severity"),("time_sensitivity","Time Sensitivity"),("crowd_level","Crowd Level")]:
            while True:
                try:
                    val = int(input(f"Enter {label} (1-10)        : ").strip())
                    if 1 <= val <= 10:
                        raw[field] = val
                        break
                    print(f"   {label} must be between 1 and 10.")
                except ValueError:
                    print(f"   Please enter a number (1-10).")
        raw["description_note"] = input("Enter Description (optional) : ").strip()
 
    # Run Pipeline
    try:
        result = run_pipeline(raw)
    except Exception as e:
        print(f"\n System error: {e}")
        result = None
    return result
 
if __name__ == "__main__":
    while True:
        try:
            cli_interface()
        except KeyboardInterrupt:
            print("System exited, Goodbye <3")
            break
        print()
        again = input("Submit another request? (yes/no): ").strip().lower()
        if again not in ("yes", "y"):
            print("Goodbye <3")
            break
 

