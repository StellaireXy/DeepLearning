
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np

def generate_a_drawing(figsize, U, V, noise=0.0):
    fig = plt.figure(figsize=(figsize,figsize))
    ax = plt.subplot(111)
    plt.axis('Off')
    ax.set_xlim(0,figsize)
    ax.set_ylim(0,figsize)
    ax.fill(U, V, "k")
    fig.canvas.draw()
    imdata = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)[::3].astype(np.float32)
    imdata = imdata + noise * np.random.random(imdata.size)
    plt.close(fig)
    return imdata

def generate_a_rectangle(noise=0.0, free_location=False):
    figsize = 1.0    
    U = np.zeros(4)
    V = np.zeros(4)
    if free_location:
        corners = np.random.random(4)
        top = max(corners[0], corners[1])
        bottom = min(corners[0], corners[1])
        left = min(corners[2], corners[3])
        right = max(corners[2], corners[3])
    else:
        side = (0.3 + 0.7 * np.random.random()) * figsize
        top = figsize/2 + side/2
        bottom = figsize/2 - side/2
        left = bottom
        right = top
    U[0] = U[1] = top
    U[2] = U[3] = bottom
    V[0] = V[3] = left
    V[1] = V[2] = right
    return generate_a_drawing(figsize, U, V, noise)

def generate_a_disk(noise=0.0, free_location=False):
    figsize = 1.0
    if free_location:
        center = np.random.random(2)
    else:
        center = (figsize/2, figsize/2)
    radius = (0.3 + 0.7 * np.random.random()) * figsize/2
    N = 50
    U = np.zeros(N)
    V = np.zeros(N)
    i = 0
    for t in np.linspace(0, 2*np.pi, N):
        U[i] = center[0] + np.cos(t) * radius
        V[i] = center[1] + np.sin(t) * radius
        i = i + 1
    return generate_a_drawing(figsize, U, V, noise)

def generate_a_triangle(noise=0.0, free_location=False):
    figsize = 1.0
    if free_location:
        U = np.random.random(3)
        V = np.random.random(3)
    else:
        size = (0.3 + 0.7 * np.random.random())*figsize/2
        middle = figsize/2
        U = (middle, middle+size, middle-size)
        V = (middle+size, middle-size, middle-size)
    imdata = generate_a_drawing(figsize, U, V, noise)
    return [imdata, [U[0], V[0], U[1], V[1], U[2], V[2]]]

def generate_dataset_classification(nb_samples, noise=0.0, free_location=False):
    # Getting im_size:
    im_size = generate_a_rectangle().shape[0]
    X = np.zeros([nb_samples,im_size])
    Y = np.zeros(nb_samples)
    print('Creating data:')
    for i in range(nb_samples):
        if i % 10 == 0:
            print(i, end=' ')
        category = np.random.randint(3)
        if category == 0:
            X[i] = generate_a_rectangle(noise, free_location)
        elif category == 1: 
            X[i] = generate_a_disk(noise, free_location)
        else:
            [X[i], V] = generate_a_triangle(noise, free_location)
        Y[i] = category
    X = (X + noise) / (255 + 2 * noise)
    return [X, Y]

def generate_test_set_classification():
    np.random.seed(42)
    [X_test, Y_test] = generate_dataset_classification(300, 20, True)
    return [X_test, Y_test]


# In[3]:


[X_train, Y_train] = generate_dataset_classification(1000, 20)


# In[8]:


plt.imshow(X_train[10].reshape(72,72), cmap='gray')


# In[9]:


from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical


# In[10]:


#Transforming to categorical data
y_train = to_categorical(Y_train)


# In[11]:


#Building the neural network
n_cols = X_train.shape[1]
model = Sequential()
#model.add(Dense(128, activation='relu', input_shape = (n_cols,)))
#model.add(Dense(3, activation='softmax'))
model.add(Dense(3, activation='softmax', input_shape = (n_cols,)))


# In[13]:


#Compiling the model
#model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[14]:


#Training the model
model.fit(X_train, y_train, nb_epoch=200, batch_size=32)


# In[ ]:


#Saving model
from keras.models import load_model
model.save('model2_file.h5')


# # Checking the classifier

# In[16]:


X_test = generate_a_disk()
plt.imshow(X_test.reshape(72,72), cmap='gray')
X_test = X_test.reshape(1, X_test.shape[0])
model.predict(X_test)


# In[19]:


[X_test, v] = generate_a_triangle()
plt.imshow(X_test.reshape(72,72), cmap='gray')
X_test = X_test.reshape(1, X_test.shape[0])
model.predict(X_test)


# In[21]:


X_test = generate_a_rectangle()
plt.imshow(X_test.reshape(72,72), cmap='gray')
X_test = X_test.reshape(1, X_test.shape[0])
model.predict(X_test)


# In[22]:


[W1, W2] = model.get_weights()


# In[23]:


W1.shape


# In[24]:


w10 = W1[:,0]
w11 = W1[:,1]
w12 = W1[:,2]


# In[25]:


plt.imshow(w10.reshape(72,72), cmap='gray')


# In[26]:


plt.imshow(w11.reshape(72,72), cmap='gray')


# In[27]:


plt.imshow(w12.reshape(72,72), cmap='gray')

