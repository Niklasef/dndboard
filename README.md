# dndboard

A physical/digital D&D tabletop for Linux.

> **Keep the physical parts of tabletop RPGs physical, while using software to make the battlefield intelligent.**

Players move real miniatures and roll real dice. The table itself is a screen that shows the map, grid, fog of war, lighting and other useful effects. A camera watches the table so the software knows where every miniature stands.

**Status:** early idea/prototype stage.

---

## Vision

- Real miniatures stay the visual representation of creatures.
- Real dice stay part of the game.
- The software augments the physical game and does not replace it.
- No digital character or monster tokens on the table. The physical miniatures take their place.

## The Setup

- **Table display:** a large horizontal 4K screen (about 43") lying flat, showing the battle map. It is calibrated so that one D&D grid square is exactly 1 inch (25.4 mm).
- **Camera:** an overhead camera looking down at the table.
- **DM display:** a second screen for the game master, with hidden information, monsters, encounter controls and debugging tools.
- **Computer:** one Linux machine running everything.
- **Foundry VTT:** the eventual engine for maps, game state, vision, fog and lighting.

## What the Table Shows

- Map and grid
- Walls and doors
- Fog of war and lighting
- Environmental effects
- Movement, range and spell overlays

## How It Works

```
Physical miniature on the table
        ↓
Camera sees it
        ↓
Software works out which miniature it is and where it stands
        ↓
Foundry is told where the miniature is
        ↓
Foundry updates vision, fog and lighting
        ↓
Table display updates
```

Foundry keeps a hidden token for each miniature, and the token follows the physical one. Foundry handles everything it already does well.

## Principles

- Linux-first.
- Never guess silently. When the software is unsure which miniature is which, it says so and lets the DM correct it.
- Keep the vision system independent of Foundry, so either can change later.
- Build it so that it can be observed and debugged.
- Start small and prove the hard parts before building the rest.

## Roadmap

1. **Prove detection:** reliably find physical miniatures in photos of a simple table.
2. **Prove identification:** tell individual miniatures apart, including near-identical ones (e.g. two goblins).
3. **Go live:** follow miniatures through a live camera view as players move them.
4. **Connect Foundry:** drive Foundry's hidden tokens from the physical miniatures.
5. **Build the real table:** the large horizontal display, the calibrated grid and the DM screen.

## Prototype Hardware

- Linux computer
- Smartphone camera
- HeroQuest miniatures and a D&D Ranger miniature

## License

TBD.
