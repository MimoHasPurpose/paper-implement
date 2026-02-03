import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import utils
from model import DSFANet
tf.config.set_visible_devices([],'GPU')

def main(X,Y,GT, diff):
    print("X.shape, Y.shape, GT.shape, diff.shape: ")
    print(X.shape, ":", Y.shape,":", GT.shape,":", diff.shape)
    train_num=2000
    max_iters=1
    lr=1e-3
    index=np.argsort(diff)
    XData=tf.convert_to_tensor(X[index[:train_num]],dtype=tf.float32)
    YData=tf.convert_to_tensor(Y[index[:train_num]],dtype=tf.float32)
    print("XData.shape: ",XData.shape)
    print("XData.shape: ",XData.shape)
    model=DSFANet(num_samples=train_num)
    optimizer=tf.keras.optimizers.SGD(learning_rate=lr)
    train_loss=np.zeros(max_iters)
    for k in range(max_iters):
        with tf.GradientTape() as tape:
            loss=model((XData,YData),training=True)
        grads=tape.gradient(loss,model.trainable_variables)
        optimizer.apply_gradients(zip(grads,model.trainable_variables))

        train_loss[k]=loss.numpy()

        if k%100==0:
            print(f"iter {k:4d}, loss is {train_loss[k]:.4f}")
        
    X_tf=tf.convert_to_tensor(X,dtype=tf.float32)
    Y_tf=tf.convert_to_tensor(Y,dtype=tf.float32)
    X_out,Y_out=X_tf,Y_tf
    for layer in model.hidden_layers:
        X_out=layer(X_out)
        Y_out=layer(Y_out)
    XTest=model.out_layer(X_out).numpy()
    YTest=model.out_layer(Y_out).numpy()
    # SFA+clustering
    X_trans,Y_trans=utils.SFA(XTest,YTest)
    diff=X_trans-Y_trans
    diff=diff/np.std(diff,axis=0)
    plt.imsave(
        "DSFAdiff.png",
        (diff**2).sum(axis=1).reshape(GT.shape),
        cmap="gray"
    )
    bin_map=KMeans(n_clusters=2).fit(
        (diff**2).sum(axis=-1,keepdims=True)
    ).labels_
    plt.imsave(
    "DSFACD.png",
    bin_map.reshape(GT.shape),
    cmap="gray"
    )
    acc1=accuracy_score(GT.reshape(-1,1)/255,bin_map)
    acc2=accuracy_score(GT.reshape(-1,1)/255,1-bin_map)
    # print("Accuracy: ",max(acc1,acc2))
    # print("both accuracy",acc1,acc2)

if __name__=="__main__":
    X,Y,GT=utils.load_dataset()
    diff=utils.cva(X=X,Y=Y)
    print(GT.shape)
    plt.imsave('CVAdiff.png',diff.reshape(GT.shape),cmap='gray')
    main(X,Y,GT,diff)

