from ortools.sat.python import cp_model
from collections import defaultdict
import random


def solve():
    model = cp_model.CpModel()

    NUM_VMS = 50000
    NUM_GROUPS = 10000
    NUM_DAYS = 7

    # -------------------------------------------------
    # Generación de datos
    # -------------------------------------------------

    vms = {}

    for vm_id in range(NUM_VMS):
        group_id = vm_id % NUM_GROUPS
        vms[f"VM{vm_id}"] = f"G{group_id}"

    print(f"VMs: {len(vms)}")
    print(f"Grupos: {NUM_GROUPS}")

    # -------------------------------------------------
    # Variables
    # -------------------------------------------------

    vm_day = {
        vm: model.NewIntVar(0, NUM_DAYS - 1, f"day_{vm}")
        for vm in vms
    }

    vm_list = list(vms.keys())

    # -------------------------------------------------
    # Agrupar VMs por grupo
    # -------------------------------------------------

    groups = defaultdict(list)

    for vm, group in vms.items():
        groups[group].append(vm)

    # -------------------------------------------------
    # Restricción:
    # Las VMs del mismo grupo deben estar en días distintos
    # -------------------------------------------------

    for group, group_vms in groups.items():

        if len(group_vms) > 1:

            model.AddAllDifferent(
                [vm_day[vm] for vm in group_vms]
            )

    # -------------------------------------------------
    # Restricción:
    # No utilizar el día 3
    # -------------------------------------------------

    for vm in vm_list:
        model.Add(vm_day[vm] != 3)

    # -------------------------------------------------
    # Resolver
    # -------------------------------------------------

    solver = cp_model.CpSolver()

    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):

        print("Asignación encontrada:\n")

        for vm in vm_list:
            print(
                f"{vm} (grupo {vms[vm]}) "
                f"-> día {solver.Value(vm_day[vm])}"
            )

    else:
        print("No existe solución")


if __name__ == "__main__":
    solve()