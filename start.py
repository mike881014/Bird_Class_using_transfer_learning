import fine_tune
import pre_train
from tool import data_loader

if __name__ == "__main__":
    while True:
        op = input("1) Pre_Train\n2) Fine_tune\n3) Predict\n4) Resize Image\n5) Exit\n\nPlease key in your option:")
        if op == "1":
            pre_train.main_start()
        elif op == "2":
            fine_tune.main_start()
        elif op == "3":
            print("\nDisable\n")
        elif op == "4":
            try:
                data_loader.re_size("picture", "target_data")
            except:
                print("\n#~ Directory Error! Please check source and target.\n")
        else:
            print("Thanks Coming!")
            break
