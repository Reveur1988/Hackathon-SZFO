Решение основано на обучении различных энкодеров через TripletLoss

1. Для установки зависимостей и виртуальной среды
   1. Run `make setup_ws`
1. Для запуска тренировки
   1. Настроить конфиг configs/train.yaml
   1. Run `make run_training`

Erorrs:

clearml-init

ModuleNotFoundError: No module named '\_libvips'
sudo apt -y install libvips-dev

sudo apt-get install python-wxtools
