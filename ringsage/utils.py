from torch_geometric.data import Batch

from ringsage.ringsageconv.cycle_detection import get_cycle_info


def cycle_collator(batch):
    batch = Batch.from_data_list(batch)
    batch.cycle_info = get_cycle_info(batch)
    return batch