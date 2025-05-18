#!/usr/bin/env python
# coding: utf-8

# # Travelling Salesman
# ## Convex hull approximate solution
# - Start with a convex hull around all points
# - Build up the path using a greedy algorithm that adding the point that is closest to an existing edge
# - Once all the points are added, tweak the path
#     - move around every edge and check whether going to the closest point shortens the overall path
#     - continue until no improvements are found
# 

# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pylab as plt
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import time
from IPython.display import clear_output


# In[45]:


def extra_peri(p1, p2, p3):
    '''calculate extra distance'''
    return np.linalg.norm((p1 - p3)) + np.linalg.norm((p2 - p3)) - np.linalg.norm((p1 - p2))

def internal(points,route):
    return np.delete(points,route,0)
    

def find_nearest(i,points,route,inside=True):
    """Find nearest point to edge defined by position i in route
    If inside is True, use all points inside the route (used to find initial route)
    Else use all points not on the current edge (used to tweak final route)"""
    min_extra = 999
    edge = (points[route[i-1]],points[route[i]])
    sample = internal(points,route) if inside else np.delete(points,[route[i-1],route[i]],axis=0)
    for point in sample:
        A = extra_peri(*edge,point)
        if A < min_extra:
            min_extra = A
            closest = point
    return (np.where(points==closest)[0])[0], min_extra
        
def plot_route(points,route,building=True):
    clear_output(wait=True)
    froute = np.append(route,route[0])
    plt.plot(points[:,0], points[:,1], 'o')
    plt.plot(points[froute,0],points[froute,1])
    #for i,(x,y) in enumerate(points):
    #    plt.text(x, y, i, va='bottom', ha='center')
    plt.title(f'Route length {route_length(route,points):.2f} {"Building" if building else "Tweaked"}')
    plt.axis('off')
    plt.savefig(f'ims/{int(time.time())}{int(time.time() % 1 * 1e6):06d}.png')

    plt.show()

    


# In[20]:



def find_best_edge(route,points):
    min_dist = 999
    best_edge = 0
    #nearest_points = {}
    for i in range(len(route)):
        (p,d) = find_nearest(i,points,route)
        #nearest_points[(route[i-1],route[i])] = (p,d)
        if d < min_dist:
            min_dist = d
            best_edge = (i,p)
    i = best_edge[0]
    #del nearest_points[(route[i-1],route[i])]
    return best_edge

def route_length(route,points):
    return sum([np.linalg.norm(points[route[i]]-points[route[i-1]]) for i in range(len(route))])
    


# In[66]:


def find_initial_route(points):
    hull = ConvexHull(points)
    route = hull.vertices
    fig,ax = plt.subplots(1,1)

    while len(route) < len(points):
        plot_route(points,route)
        best_edge = find_best_edge(route,points)
        #print('Best edge',best_edge)#,route,nearest_points)
        #print(points[10],'Internal',internal(points,route))

        route = np.insert(route, best_edge[0], best_edge[1])
        #time.sleep(0.5)

    plot_route(points,route)
    return route


def salesman(points):
    route = find_initial_route(points)
    ### Tweak initial route
    route = tweak_route(route,points)
    
    
    return route


# In[ ]:





# In[ ]:





# - Tweak route
#     - The idea is to find the closest point to each edge and check whether visiting it between the two end points of the edge reduces the total path length
# 

# In[72]:


def tweak_route(route,points):
    """The idea is to find the closest point to each edge and 
    check whether visiting it between the two end points of the edge 
    reduces the total path length"""
    
    shortest_route = route
    shortest_length = route_length(shortest_route,points)
    improvements = [shortest_length]
    found = True
    while found:
        for i in range(len(route)):
            route2 = route
            candidate,_ = find_nearest(i,points,route,inside=False)
            end_point = route[i]
            index_c = np.where(route2==candidate)[0][0]
            route2 = np.delete(route2,index_c)
            index_to = np.where(route2==end_point)[0][0]
            route2 = np.insert(route2, index_to, candidate)

            if route_length(route2,points) < shortest_length:
                shortest_route = route2
                shortest_length = route_length(shortest_route,points)
                improvements += [shortest_length]
                break
        found = not np.array_equal(route,shortest_route) 
        route = shortest_route
        #time.sleep(0.5)
        plot_route(points,route,False)
    print(f"Improvements from tweaking initial route: {['{0:.2f}'.format(i) for i in improvements]}")
    return route          


# In[75]:


#n = 300
#points = np.random.random((n, 2) )  # 30 random points in 2-D


# In[76]:


#route = salesman(points)
#plot_route(points,route)


# In[ ]:





# In[ ]:





# In[ ]:




