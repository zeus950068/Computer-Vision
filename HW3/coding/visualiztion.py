import matplotlib.pyplot as plt
import numpy as np

x_train_dataloader = 1.0
x_half_train_dataloader = 0.5
x_sixteenth_train_dataloader = 1/16
data_size = np.array([x_sixteenth_train_dataloader, x_half_train_dataloader, x_train_dataloader]) 

# ########## with no weights(256 batches) #############
# batch size = 256
# small_model_accuracy_train = np.array([0.02404, 0.3117, 0.69926]) 
# small_model_accuracy_test = np.array([0.3687, 0.6349, 0.7118]) 
# big_model_accuracy_train =  np.array([0.01772, 0.24574, 0.56702]) 
# big_model_accuracy_test =  np.array([0.3061, 0.4215, 0.5003])

# plt.xlabel("Dataset size")
# plt.ylabel("Accuracy")
# plt.plot(data_size, small_model_accuracy_train, color = 'royalblue', marker = 'o', label = 'Small Model')
# plt.plot(data_size, big_model_accuracy_train, color = 'darkorange', marker = 'o', label = 'Big Model')
# plt.legend(loc='lower right')
# plt.title("Train Accuracy with weights 'None'(batch size = 256)")
# plt.savefig('Train Accuracy with weights None(batch size = 256).png')
# plt.show()
# print("\n\n")

# plt.xlabel("Dataset size")
# plt.ylabel("Accuracy")
# plt.plot(data_size, small_model_accuracy_test, color = 'royalblue', marker = 'o', label = 'Small Model')
# plt.plot(data_size, big_model_accuracy_test, color = 'darkorange', marker = 'o', label = 'Big Model')
# plt.legend(loc='lower right')
# plt.title("Testing Accuracy with weights 'None'(batch size = 256)")
# plt.savefig('Testing Accuracy with weights None(batch size = 256).png')
# plt.show()

# ############ with no weights(128 batches) #############
# # batch size = 128
# small_model_accuracy_train = np.array([0.02538, 0.3176, 0.7063]) 
# small_model_accuracy_test = np.array([0.4416, 0.6552, 0.7151]) 
# big_model_accuracy_train =  np.array([0.01888, 0.21388, 0.4715]) 
# big_model_accuracy_test =  np.array([0.3397, 0.4486, 0.321])

# plt.xlabel("Dataset size")
# plt.ylabel("Accuracy")
# plt.plot(data_size, small_model_accuracy_train, color = 'royalblue', marker = 'o', label = 'Small Model')
# plt.plot(data_size, big_model_accuracy_train, color = 'darkorange', marker = 'o', label = 'Big Model')
# plt.legend(loc='lower right')
# plt.title("Train Accuracy with weights 'None'(batch size = 128)")
# plt.savefig('Train Accuracy with weights None (batch size = 128).png')
# plt.show()
# print("\n\n")

# plt.xlabel("Dataset size")
# plt.ylabel("Accuracy")
# plt.plot(data_size, small_model_accuracy_test, color = 'royalblue', marker = 'o', label = 'Small Model')
# plt.plot(data_size, big_model_accuracy_test, color = 'darkorange', marker = 'o', label = 'Big Model')
# plt.legend(loc='lower right')
# plt.title("Testing Accuracy with weights 'None'(batch size = 128)")
# plt.savefig('Testing Accuracy with weights None (batch size = 128).png')
# plt.show()

# ############ with no weights(64 batches) #############
# # batch size = 64
# small_model_accuracy_train = np.array([0.0249, 0.31468, 0.706]) 
# small_model_accuracy_test = np.array([0.3924, 0.646, 0.7318]) 
# big_model_accuracy_train =  np.array([0.01842, 0.16418, 0.51084]) 
# big_model_accuracy_test =  np.array([0.2732, 0.2552, 0.4477])

# plt.xlabel("Dataset size")
# plt.ylabel("Accuracy")
# plt.plot(data_size, small_model_accuracy_train, color = 'royalblue', marker = 'o', label = 'Small Model')
# plt.plot(data_size, big_model_accuracy_train, color = 'darkorange', marker = 'o', label = 'Big Model')
# plt.legend(loc='lower right')
# plt.title("Train Accuracy with weights 'None'(batch size = 64)")
# plt.savefig('Train Accuracy with weights None (batch size = 64).png')
# plt.show()
# print("\n\n")

# plt.xlabel("Dataset size")
# plt.ylabel("Accuracy")
# plt.plot(data_size, small_model_accuracy_test, color = 'royalblue', marker = 'o', label = 'Small Model')
# plt.plot(data_size, big_model_accuracy_test, color = 'darkorange', marker = 'o', label = 'Big Model')
# plt.legend(loc='lower right')
# plt.title("Testing Accuracy with weights 'None'(batch size = 64)")
# plt.savefig('Testing Accuracy with weights None (batch size = 64).png')
# plt.show()


###################################################################################################################
###################################################################################################################
###################################################################################################################

# ############ with weights(256 batches) #############
# # batch size = 256
# small_model_accuracy_train = np.array([0.03884, 0.39472, 0.82326]) 
# small_model_accuracy_test = np.array([0.6395, 0.7802, 0.7989]) 
# big_model_accuracy_train =  np.array([0.04068, 0.38906, 0.8223]) 
# big_model_accuracy_test =  np.array([0.6709, 0.798,  0.8152])

# plt.xlabel("Dataset size")
# plt.ylabel("Accuracy")
# plt.plot(data_size, small_model_accuracy_train, color = 'royalblue', marker = 'o', label = 'Small Model')
# plt.plot(data_size, big_model_accuracy_train, color = 'darkorange', marker = 'o', label = 'Big Model')
# plt.legend(loc='lower right')
# plt.title("Train Accuracy with weights 'IMAGENET1K V1(batch size = 256)'")
# plt.savefig('Train Accuracy with weights IMAGENET1K V1(batch size = 256).png')
# plt.show()
# print("\n\n")

# plt.xlabel("Dataset size")
# plt.ylabel("Accuracy")
# plt.plot(data_size, small_model_accuracy_test, color = 'royalblue', marker = 'o', label = 'Small Model')
# plt.plot(data_size, big_model_accuracy_test, 'darkorange', marker = 'o', label = 'Big Model')
# plt.legend(loc='lower right')
# plt.title("Testing Accuracy with weights 'IMAGENET1K V1(batch size = 256)'")
# plt.savefig('Testing Accuracy with weights IMAGENET1K V1(batch size = 256).png')
# plt.show()

# # ############ with weights(128 batches) #############
# # batch size = 128
# small_model_accuracy_train = np.array([0.0399,  0.38708, 0.81038]) 
# small_model_accuracy_test = np.array([0.6408, 0.7844, 0.7958]) 
# big_model_accuracy_train =  np.array([0.04106, 0.37676, 0.776]) 
# big_model_accuracy_test =  np.array([0.6475, 0.7486, 0.7094])

# plt.xlabel("Dataset size")
# plt.ylabel("Accuracy")
# plt.plot(data_size, small_model_accuracy_train, color = 'royalblue', marker = 'o', label = 'Small Model')
# plt.plot(data_size, big_model_accuracy_train, color = 'darkorange', marker = 'o', label = 'Big Model')
# plt.legend(loc='lower right')
# plt.title("Train Accuracy with weights 'IMAGENET1K V1'(batch size = 128)")
# plt.savefig('Train Accuracy with weights IMAGENET1K V1(batch size = 128).png')
# plt.show()
# print("\n\n")

# plt.xlabel("Dataset size")
# plt.ylabel("Accuracy")
# plt.plot(data_size, small_model_accuracy_test, color = 'royalblue', marker = 'o', label = 'Small Model')
# plt.plot(data_size, big_model_accuracy_test, 'darkorange', marker = 'o', label = 'Big Model')
# plt.legend(loc='lower right')
# plt.title("Testing Accuracy with weights 'IMAGENET1K V1'(batch size = 128)")
# plt.savefig('Testing Accuracy with weights IMAGENET1K V1(batch size = 128).png')
# plt.show()

# # ############ with weights(64 batches) #############
# # batch size = 64
# small_model_accuracy_train = np.array([0.03814, 0.3741, 0.78714]) 
# small_model_accuracy_test = np.array([0.571, 0.7734, 0.7786]) 
# big_model_accuracy_train =  np.array([0.03634, 0.35632, 0.75314]) 
# big_model_accuracy_test =  np.array([0.6276, 0.7508, 0.5983])

# plt.xlabel("Dataset size")
# plt.ylabel("Accuracy")
# plt.plot(data_size, small_model_accuracy_train, color = 'royalblue', marker = 'o', label = 'Small Model')
# plt.plot(data_size, big_model_accuracy_train, color = 'darkorange', marker = 'o', label = 'Big Model')
# plt.legend(loc='lower right')
# plt.title("Train Accuracy with weights 'IMAGENET1K V1'(batch size = 64)")
# plt.savefig('Train Accuracy with weights IMAGENET1K V1(batch size = 64).png')
# plt.show()
# print("\n\n")

# plt.xlabel("Dataset size")
# plt.ylabel("Accuracy")
# plt.plot(data_size, small_model_accuracy_test, color = 'royalblue', marker = 'o', label = 'Small Model')
# plt.plot(data_size, big_model_accuracy_test, 'darkorange', marker = 'o', label = 'Big Model')
# plt.legend(loc='lower right')
# plt.title("Testing Accuracy with weights 'IMAGENET1K V1'(batch size = 64)")
# plt.savefig('Testing Accuracy with weights IMAGENET1K V1(batch size = 64).png')
# plt.show()

# ############ with weights(32 batches, lr=1e-4) #############
# small_model_accuracy_train = np.array([0.03602, 0.37854, 0.7991]) 
# small_model_accuracy_test = np.array([0.6242, 0.792, 0.8263]) 
# big_model_accuracy_train =  np.array([0.03838, 0.39822, 0.84284]) 
# big_model_accuracy_test =  np.array([0.6368, 0.828, 0.8621])

# plt.xlabel("Dataset size")
# plt.ylabel("Accuracy")
# plt.plot(data_size, small_model_accuracy_train, color = 'royalblue', marker = 'o', label = 'Small Model')
# plt.plot(data_size, big_model_accuracy_train, color = 'darkorange', marker = 'o', label = 'Big Model')
# plt.legend(loc='lower right')
# plt.title("Train Accuracy with weights 'IMAGENET1K V1'(batch size = 32)")
# plt.savefig('Train Accuracy with weights IMAGENET1K V1(batch size = 32).png')
# plt.show()
# print("\n\n")

# plt.xlabel("Dataset size")
# plt.ylabel("Accuracy")
# plt.plot(data_size, small_model_accuracy_test, color = 'royalblue', marker = 'o', label = 'Small Model')
# plt.plot(data_size, big_model_accuracy_test, 'darkorange', marker = 'o', label = 'Big Model')
# plt.legend(loc='lower right')
# plt.title("Testing Accuracy with weights 'IMAGENET1K V1'(batch size = 32)")
# plt.savefig('Testing Accuracy with weights IMAGENET1K V1(batch size = 32).png')
# plt.show()

# ############ with weights(32 batches, lr=1e-3) #############
small_model_accuracy_train = np.array([0.03454, 0.3527, 0.77]) 
small_model_accuracy_test = np.array([0.6197, 0.7441, 0.7869]) 
big_model_accuracy_train =  np.array([0.0315, 0.29898, 0.67464]) 
big_model_accuracy_test =  np.array([0.5094, 0.6607, 0.7085])

plt.xlabel("Dataset size")
plt.ylabel("Accuracy")
plt.plot(data_size, small_model_accuracy_train, color = 'royalblue', marker = 'o', label = 'Small Model')
plt.plot(data_size, big_model_accuracy_train, color = 'darkorange', marker = 'o', label = 'Big Model')
plt.legend(loc='lower right')
plt.title("Train Accuracy with weights 'IMAGENET1K V1'(batch size = 32)")
plt.savefig('Train Accuracy with weights IMAGENET1K V1(batch size = 32).png')
plt.show()
print("\n\n")

plt.xlabel("Dataset size")
plt.ylabel("Accuracy")
plt.plot(data_size, small_model_accuracy_test, color = 'royalblue', marker = 'o', label = 'Small Model')
plt.plot(data_size, big_model_accuracy_test, 'darkorange', marker = 'o', label = 'Big Model')
plt.legend(loc='lower right')
plt.title("Testing Accuracy with weights 'IMAGENET1K V1'(batch size = 32)")
plt.savefig('Testing Accuracy with weights IMAGENET1K V1(batch size = 32).png')
plt.show()