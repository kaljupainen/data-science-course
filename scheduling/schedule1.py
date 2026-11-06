from ortools.sat.python import cp_model
import json


def solve():
    model = cp_model.CpModel()

    NUM_VMS = 500
    NUM_GROUPS = 100
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

    # Máquinas virtuales y grupo al que pertenecen

    days = range(NUM_DAYS)

    # Variable: día asignado a cada VM
    # Cada VM podrá tomar cualquiera de estos días como valor (en caso de que sean diferentes las máquinas de pre y pro, 
    # los rangos son diferentes)
    vm_day = {
        vm: model.NewIntVar(0, NUM_DAYS - 1, f"day_{vm}")
        for vm in vms
    }

    # print(vm_day)

    # Restricción:
    # VMs del mismo grupo no pueden coincidir en el mismo día
    vm_list = list(vms.keys())

    for i in range(len(vm_list)):
        for j in range(i + 1, len(vm_list)):
            vm1 = vm_list[i]
            vm2 = vm_list[j]

            # Si los grupos coinciden, se añade la restricción 
            if vms[vm1] == vms[vm2]:
                model.Add(vm_day[vm1] != vm_day[vm2])

            # Si el día 3 es fin de semana, no usarlo (por ejemplo, if maquina es pro:)
            model.Add(vm_day[vm1] != 3)

    # Resolver
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