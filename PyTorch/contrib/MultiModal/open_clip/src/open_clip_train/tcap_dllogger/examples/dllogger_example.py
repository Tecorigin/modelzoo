from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity

json_logger = Logger(
[
    StdOutBackend(Verbosity.DEFAULT),
    JSONStreamBackend(Verbosity.VERBOSE, 'dlloger_example.json'),
]
)

json_logger.metadata("train.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.loss_mean", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("val.loss", {"unit": "", "GOAL": "MINIMIZE", "STAGE": "VALID"})
json_logger.metadata("train.ips",{"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("val.ips",{"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "VALID"})
json_logger.metadata("train.compute_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.fp_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.bp_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})
json_logger.metadata("train.grad_time", {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"})


# json_logger.metadata_batch(
#     ["train.ips","train.ips_sum", "val.ips",],
#     {"unit": "imgs/s", "format": ":.3f", "GOAL": "MAXIMIZE", "STAGE": "TRAIN"},
# )
# json_logger.metadata_batch(
#             ["train.compute_time", "train.fp_time", "train.bp_time", "train.grad_time"],
#             {"unit": "s", "GOAL": "MINIMIZE", "STAGE": "TRAIN"}
#             )

global_step = 0
for epo in range(4):
    # train_dataloader
    for idx in range(3):
        ## make loss reduction
        # reduced_loss = reduce_loss(loss, is_dict=False, mode="mean")
        # reduced_ips = reduce_loss(ips, is_dict=False, mode="sum")
        reduced_loss = 0.003
        reduced_ips = 20.0

        json_logger.log(
            step = (epo, global_step),
            data = {
                    "rank":0,
                    "train.loss":0.01, 
                    "train.ips":3.0,
                    "data.shape":[16,3,224,224],
                    "train.lr":0.0001,
                    "train.data_time":1.03,
                    "train.compute_time":0.003,
                    "train.fp_time":3.02,
                    "train.bp_time":6.02,
                    "train.grad_time":0.003,
                    },
            verbosity=Verbosity.DEFAULT,
        )

        # if True: #is_main_process()
        #     # Epoch Iteration rank: master, train.loss_mean train.ips_sum
        #     json_logger.log(
        #         step = (epo, global_step),
        #         data = {
        #                 "rank":"master",
        #                 "train.loss_mean":reduced_loss, 
        #                 "train.ips_sum":reduced_ips,
        #                 },
        #         verbosity=Verbosity.DEFAULT,
        #     )

        global_step += 1

    # val_loader
    for index in range(2):
        # make evaluation
        pass

    json_logger.log(
        step = (epo, global_step),
        data = {
            "val.loss":0.002,
            "val.ips":5.00,
            "val.top1":0.54
        },
        verbosity=Verbosity.DEFAULT,
    )

