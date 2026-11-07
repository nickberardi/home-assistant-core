"""Handlers for UniFi Network Integration API data that needs one request per device.

The device list of the Integration API carries no ports and no statistics,
so these handlers fetch them per device and keep them like aiounifi keeps
listed items, signalling added, changed and deleted objects.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import NoReturn, override

import aiounifi
from aiounifi.interfaces.api_handlers import ItemEvent
from aiounifi.models.api import ApiItem
from aiounifi.network.v1.interfaces.api_handlers import APIHandler
from aiounifi.network.v1.models.device import DevicePort, DeviceState, DeviceStatistics

from ..const import LOGGER

# Ports and statistics take one request per device
MAX_PARALLEL_REQUESTS = 4

OBJ_ID = "obj_id"


class NetworkPort(ApiItem):
    """A PoE capable port of a device, keyed by MAC address and port index."""

    raw: DevicePort

    @property
    def idx(self) -> int:
        """Port index."""
        return self.raw["idx"]

    @property
    def poe_enabled(self) -> bool:
        """Whether PoE is enabled on the port."""
        return self.raw.get("poe", {}).get("enabled", False)


class _DerivedHandler[ItemT: ApiItem](APIHandler[ItemT]):
    """Handler filled from per device requests instead of a list endpoint."""

    obj_id_key = OBJ_ID

    def __init__(self, api: aiounifi.Controller) -> None:
        """Initialize."""
        super().__init__(api.network)
        self.devices = api.network.devices
        self.devices.subscribe(
            self._device_changed, (ItemEvent.CHANGED, ItemEvent.DELETED)
        )

    def _device_changed(self, event: ItemEvent, mac: str) -> None:
        """Follow the device the objects belong to, which is polled more often.

        Objects of a removed device are dropped, the others are signalled as
        their availability follows the device state.
        """
        for obj_id in [obj_id for obj_id in self if obj_id.partition("_")[0] == mac]:
            if event is ItemEvent.DELETED:
                self._forget(obj_id)
            else:
                self.signal_subscribers(ItemEvent.CHANGED, obj_id)

    @override
    def list_request(self, offset: int, limit: int) -> NoReturn:
        """Derived data has no list endpoint."""
        raise NotImplementedError

    async def _fetch_per_device[T](
        self, macs: list[str], fetch: Callable[[str], Awaitable[T]]
    ) -> dict[str, T]:
        """Fetch data of each device with a few requests in flight.

        A device can fail on its own, for example when it was just removed,
        so only a failure of every request is raised.
        """
        semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)

        async def fetch_one(mac: str) -> T:
            async with semaphore:
                return await fetch(self.devices[mac].device_id)

        results = await asyncio.gather(
            *(fetch_one(mac) for mac in macs), return_exceptions=True
        )
        fetched: dict[str, T] = {}
        errors: list[BaseException] = []
        for mac, result in zip(macs, results, strict=True):
            if isinstance(result, BaseException):
                LOGGER.debug("Could not fetch UniFi device %s data: %r", mac, result)
                errors.append(result)
            else:
                fetched[mac] = result
        if errors and not fetched:
            raise errors[0]
        return fetched


class NetworkPorts(_DerivedHandler[NetworkPort]):
    """PoE capable ports of the adopted devices.

    Ports only change with the hardware, so the details of each device are
    fetched once.
    """

    item_cls = NetworkPort

    def __init__(self, api: aiounifi.Controller) -> None:
        """Initialize."""
        super().__init__(api)
        self._ports: dict[str, list[DevicePort]] = {}

    @override
    async def update(self) -> None:
        """Fetch the ports of devices seen for the first time."""
        for mac in [mac for mac in self._ports if mac not in self.devices]:
            del self._ports[mac]
        new = [
            mac
            for mac, device in self.devices.items()
            if mac not in self._ports and device.supported
        ]
        details = await self._fetch_per_device(new, self.devices.get_details)
        for mac, device in details.items():
            self._ports[mac] = [port for port in device.ports if "poe" in port]
        self.reconcile(
            [
                {**port, OBJ_ID: f"{mac}_{port['idx']}"}
                for mac, ports in self._ports.items()
                for port in ports
            ]
        )


class NetworkStatistics(_DerivedHandler[DeviceStatistics]):
    """Latest statistics of the online devices.

    A device that goes offline keeps its last statistics, its sensors tell
    from the device state that they are not current.
    """

    item_cls = DeviceStatistics
    keep_missing = True

    @override
    async def update(self) -> None:
        """Fetch the statistics of every online device."""
        online = [
            mac
            for mac, device in self.devices.items()
            if device.state is DeviceState.ONLINE
        ]
        statistics = await self._fetch_per_device(online, self.devices.get_statistics)
        self.reconcile(
            [{**stats.raw, OBJ_ID: mac} for mac, stats in statistics.items()]
        )
