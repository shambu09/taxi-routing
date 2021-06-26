import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from testServer import Server
from log import Logger
from queue import Queue

Logger.clear()
log = Logger.log

q = Queue()
data = np.load("data/passengers.npz", allow_pickle=True)

n = data["p_num"]
pos = data["p_pos"]
dest = data["p_dest"]
driver = data["d_pos"].reshape((1, 2))

training1 = np.concatenate([pos, dest], axis=1)
training2 = np.concatenate([dest, pos], axis=1)
training = np.concatenate([training1, training2], axis=0)

assert (training.shape[1] == 4)
kmeans = KMeans(n_clusters=10, random_state=0).fit(training)

p_d = np.concatenate([pos, dest], axis=0)
color = np.concatenate([np.zeros(n), np.ones(n)], axis=0)
text = kmeans.labels_
fig = px.scatter(
    x=p_d[:, 0],
    y=p_d[:, 1],
    color=text,
    hover_name=["Start" if i == 0 else "Destination" for i in color])

classes = kmeans.predict(training1).reshape((-1, 1))

log(training1.shape)
log(classes.shape)

curr_center = np.concatenate(
    (kmeans.cluster_centers_.squeeze(), np.arange(0, 10).reshape(
        10, 1), np.ones((10, 1))),
    axis=1)

curr_passengers = np.concatenate(
    (training1[:, 0:2], np.ones((training1.shape[0], 1))), axis=1)


#* get passengers with same class
def getSameClass(_class, classes, data):
    data = data[data[:, -1] == 1]
    mask = classes == _class
    print(mask.squeeze().shape)
    print(data.shape)
    return data[mask.squeeze(), :]


#* get nearest center and remove it from the list
def getNearestCentreClass(centers, driver):
    mask = centers[:, -1] == 1
    tmp = centers[mask]

    dis = np.linalg.norm(tmp[:, 0:2].copy() - driver)
    mini = np.argmin(dis)
    _class = tmp[mini, 4]
    tmp[mini, 5] = 0
    centers[mask] = tmp
    return _class, centers


#* get nearest passenger in the same class
def getNearestPassenger(driver, passengers):
    mask = passengers[:, -1] == 1
    tmp = passengers[mask]
    # print(passengers.shape)
    # print(tmp.shape)

    dis = np.linalg.norm(tmp[:,0:2].copy() - driver)
    mini = np.argmin(dis)
    tmp[mini, 2] = 0
    passengers[mask] = tmp
    return passengers[mini, 0:2], passengers, tmp.shape[0] - 1


log("\n")
log("centers")
assert (len(curr_center) == 10)
log(curr_center)
log(curr_center.shape)

#check driver belongs to which  cluster
driver = np.array([0, 0])
assert (driver.shape == curr_passengers[0, 0:2].shape)

queue = []
d_class, curr_center = getNearestCentreClass(curr_center, driver)
tmp_passengers = getSameClass(d_class, classes, curr_passengers)
traversed = 1
k = 1

f = 0
#------------------------------------------------------#
#*taxi starting!
while (traversed <= 10):
    f += 1
    if f == 1000:
        print("Stack limit")
        break

    while (k != 0):
        tmp_driver, tmp_passengers, k = getNearestPassenger(
            driver, tmp_passengers)
        queue.append((driver, tmp_driver))
        driver = tmp_driver

    d_class, curr_center = getNearestCentreClass(curr_center, driver)
    tmp_passengers = getSameClass(d_class, classes, tmp_passengers)
    k = 1
    traversed += 1

log("\n\n")
log(queue)
#-------------------------------------------------------#

if __name__ == '__main__':
    # app = Server()
    # app(fig)
    # app.run()
    pass